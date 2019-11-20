use rustpython_parser::ast as pyast;
use rustpython_parser::location::Location;
use thiserror::Error;

use parity_wasm::builder::{self, ModuleBuilder};
use parity_wasm::elements::{self, Instruction, Module, ValueType};

use num_traits::ToPrimitive;
use std::collections::{hash_map, HashMap};

#[derive(Error, Debug)]
pub enum Error {
    #[error("error parsing python code: {0}")]
    ParsingError(#[from] rustpython_parser::error::ParseError),
    #[error("unsupported python statement at {0}")]
    UnsupportedStatement(Location),
    #[error("unsupported arg types in function at {0}")]
    UnsupportedArgs(Location),
    #[error("unknown wasm type at {0}")]
    UnknownType(Location),
    #[error("redeclaration of of name '{0}' at {1}")]
    Redecl(String, Location),
    #[error("could not transform number literal to WASM type at {0}")]
    BadNumber(Location),
    #[error("type mismatch at {0}")]
    TypeMismatch(Location),
    #[error("unknown name '{0}' at {1}")]
    UnknownIdentifier(String, Location),
    #[error("can't use void type as result of expression at {0}")]
    UnexpectedVoid(Location),
    #[error("{0} at {1}")]
    Other(&'static str, Location),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, PartialEq)]
enum AsmPyType {
    Primitive(Primitive),
    Void,
}

#[derive(Clone, Copy, PartialEq)]
enum Primitive {
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
}

impl AsmPyType {
    fn to_value_type(&self) -> Option<ValueType> {
        match self {
            AsmPyType::Primitive(Primitive::I32) => Some(ValueType::I32),
            AsmPyType::Primitive(Primitive::I64) => Some(ValueType::I64),
            AsmPyType::Primitive(Primitive::U32) => Some(ValueType::I32),
            AsmPyType::Primitive(Primitive::U64) => Some(ValueType::I64),
            AsmPyType::Primitive(Primitive::F32) => Some(ValueType::F32),
            AsmPyType::Primitive(Primitive::F64) => Some(ValueType::F64),
            AsmPyType::Void => None,
        }
    }

    fn try_to_value_type(&self, loc: &Location) -> Result<ValueType> {
        self.to_value_type()
            .ok_or_else(|| Error::UnexpectedVoid(loc.clone()))
    }
}

#[derive(Clone)]
struct FuncTypeSig {
    params: Vec<AsmPyType>,
    ret: AsmPyType,
}

macro_rules! sig_convert {
    ($sig:expr) => {{
        let sig: &FuncTypeSig = &$sig;
        builder::signature()
            .with_params(
                sig.params
                    .iter()
                    .map(|ty| ty.to_value_type().unwrap())
                    .collect(),
            )
            .with_return_type(sig.ret.to_value_type())
            .build_sig()
    }};
}

pub fn compile(prog: pyast::Program) -> Result<Module> {
    let mut compiler = Compiler {
        module: ModuleBuilder::new(),
        items: Default::default(),
        cur_fno: 0,
    };
    compiler.inject_program(prog)?;
    compiler.compile_items()?;
    Ok(compiler.module.build())
}

struct Compiler {
    module: ModuleBuilder,
    items: HashMap<String, AsmPyItem>,
    cur_fno: u32,
}

fn primitive_from_str(s: &str) -> Option<Primitive> {
    match s {
        "i32" | "int" => Some(Primitive::I32),
        "i64" | "long" => Some(Primitive::I64),
        "u32" => Some(Primitive::U32),
        "u64" => Some(Primitive::U64),
        "f32" | "float" => Some(Primitive::F32),
        "f64" | "double" => Some(Primitive::F64),
        _ => None,
    }
}

fn simple_ty_from_str(s: &str) -> Option<AsmPyType> {
    primitive_from_str(s).map(AsmPyType::Primitive)
}

impl Compiler {
    fn lookup_type(&self, ty: &pyast::Expression) -> Option<AsmPyType> {
        match &ty.node {
            pyast::ExpressionType::Identifier { name } => {
                simple_ty_from_str(name).or_else(|| match self.items.get(name) {
                    Some(AsmPyItem::TypeAlias(ty)) => Some(ty.clone()),
                    _ => None,
                })
            }
            _ => None,
        }
    }

    fn try_lookup_type(&self, ty: &pyast::Expression) -> Result<AsmPyType> {
        self.lookup_type(ty)
            .ok_or_else(|| Error::UnknownType(ty.location.clone()))
    }

    fn fno(&mut self) -> u32 {
        let fno = self.cur_fno;
        self.cur_fno += 1;
        fno
    }

    fn inject_program(&mut self, module: pyast::Program) -> Result<()> {
        for item in module.statements {
            self.inject_item(item)?;
        }

        Ok(())
    }

    fn inject_item(&mut self, stmt: pyast::Statement) -> Result<()> {
        let loc = stmt.location;
        match stmt.node {
            pyast::StatementType::FunctionDef {
                is_async,
                name,
                args,
                body,
                decorator_list,
                returns,
            } => {
                if is_async {
                    return Err(Error::UnsupportedStatement(loc));
                }
                if !args.kwonlyargs.is_empty()
                    || args.vararg != pyast::Varargs::None
                    || args.kwarg != pyast::Varargs::None
                    || !args.defaults.is_empty()
                    || !args.kw_defaults.is_empty()
                    || !decorator_list.is_empty()
                {
                    return Err(Error::UnsupportedArgs(loc));
                }
                let mut param_names = Vec::with_capacity(args.args.len());
                let mut param_types = Vec::with_capacity(param_names.len());
                for arg in &args.args {
                    let t = match &arg.annotation {
                        Some(t) => t,
                        None => return Err(Error::UnknownType(arg.location.clone())),
                    };
                    let ty = self.try_lookup_type(t)?;
                    param_names.push(arg.arg.clone());
                    param_types.push(ty);
                }
                let ret = match returns {
                    Some(ref ty) => self.try_lookup_type(ty)?,
                    None => AsmPyType::Void,
                };
                let sig = FuncTypeSig {
                    params: param_types,
                    ret,
                };
                let fno = self.fno();
                match self.items.entry(name) {
                    hash_map::Entry::Occupied(o) => Err(Error::Redecl(o.key().clone(), loc)),
                    hash_map::Entry::Vacant(v) => {
                        v.insert(AsmPyItem::Function(AsmPyFunction {
                            params: param_names,
                            sig,
                            body,
                            export: true, // TODO
                            fno,
                        }));
                        Ok(())
                    }
                }
            }
            pyast::StatementType::Assign { targets, value } => match value.node {
                pyast::ExpressionType::Call {
                    function,
                    args,
                    keywords,
                } => {
                    let decorator_name = match function.node {
                        pyast::ExpressionType::Identifier { name } => name,
                        _ => return Err(Error::UnsupportedStatement(function.location)),
                    };
                    for target in targets {
                        match target.node {
                            pyast::ExpressionType::Identifier { name } => self
                                .inject_assign_decorator(
                                    name,
                                    &decorator_name,
                                    &args,
                                    &keywords,
                                    value.location.clone(),
                                )?,
                            _ => return Err(Error::UnsupportedStatement(value.location)),
                        }
                    }
                    Ok(())
                }
                _ => Err(Error::UnsupportedStatement(value.location)),
            },
            _ => Err(Error::UnsupportedStatement(loc)),
        }
    }

    fn inject_assign_decorator(
        &mut self,
        item_name: String,
        name: &str,
        args: &[pyast::Expression],
        _kwargs: &[pyast::Keyword],
        loc: pyast::Location,
    ) -> Result<()> {
        match name {
            "extern_func" => {
                let (ns, name, params, ret) = match args {
                    [a, b, c, d] => (a, b, c, Some(d)),
                    [a, b, c] => (a, b, c, None),
                    _ => return Err(Error::UnsupportedStatement(loc)),
                };
                let ns = match &ns.node {
                    pyast::ExpressionType::String { value } => match value {
                        pyast::StringGroup::Constant { value } => Some(value.clone()),
                        _ => None,
                    },
                    _ => None,
                }
                .ok_or_else(|| Error::Other("namespace must be a string", ns.location.clone()))?;
                let name = match &name.node {
                    pyast::ExpressionType::String { value } => match value {
                        pyast::StringGroup::Constant { value } => Some(value.clone()),
                        _ => None,
                    },
                    _ => None,
                }
                .ok_or_else(|| Error::Other("name must be a string", name.location.clone()))?;
                let params = match &params.node {
                    pyast::ExpressionType::Tuple { elements } => elements
                        .iter()
                        .map(|expr| self.try_lookup_type(expr))
                        .collect::<Result<Vec<_>>>()?,
                    _ => {
                        return Err(Error::Other(
                            "params must be a tuple of types",
                            params.location.clone(),
                        ))
                    }
                };
                let ret = match ret {
                    Some(ref ty) => self.try_lookup_type(ty)?,
                    None => AsmPyType::Void,
                };
                let sig = FuncTypeSig { params, ret };
                let func_ty = self.module.push_signature(sig_convert!(&sig));
                self.module.push_import(elements::ImportEntry::new(
                    ns,
                    name,
                    elements::External::Function(func_ty),
                ));
                let fno = self.fno();
                self.items
                    .insert(item_name, AsmPyItem::ExternFunction { sig, fno });
                Ok(())
            }
            _ => Err(Error::UnsupportedStatement(loc)),
        }
    }

    fn compile_items(&mut self) -> Result<()> {
        for (name, item) in &self.items {
            match item {
                AsmPyItem::Function(ref func) => {
                    let code = FunctionCompiler::new(&self, func).compile()?;
                    self.module.push_function(code);
                    if func.export {
                        self.module.push_export(elements::ExportEntry::new(
                            name.clone(),
                            elements::Internal::Function(func.fno),
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

struct FunctionCompiler<'c, 'f> {
    compiler: &'c Compiler,
    func: &'f AsmPyFunction,
    vars: HashMap<String, (AsmPyType, u32)>,
    blockdepth: u32,
    vardecls: Vec<elements::Local>,
    varno: u32,
    instructions: elements::Instructions,
    fbuilder: builder::FunctionBuilder,
}

impl<'c, 'f> FunctionCompiler<'c, 'f> {
    fn new(compiler: &'c Compiler, func: &'f AsmPyFunction) -> Self {
        let mut varno = 0u32;
        let vars = func
            .params
            .iter()
            .zip(func.sig.params.iter())
            .map(|(name, ty)| {
                let i = varno;
                varno += 1;
                (name.clone(), (ty.clone(), i))
            })
            .collect();
        FunctionCompiler {
            compiler,
            func,
            vars,
            blockdepth: 0,
            vardecls: Vec::new(),
            varno,
            instructions: elements::Instructions::new(Vec::new()),
            fbuilder: builder::function(),
        }
    }

    fn emit(&mut self, instr: Instruction) {
        self.instructions.elements_mut().push(instr);
    }

    fn compile(mut self) -> Result<builder::FunctionDefinition> {
        self.compile_function()?;
        Ok(self.finish())
    }

    fn compile_function(&mut self) -> Result<()> {
        for stmt in &self.func.body {
            self.compile_stmt(stmt)?;
        }
        self.emit(Instruction::End);
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &pyast::Statement) -> Result<()> {
        match &stmt.node {
            pyast::StatementType::Return { value } => {
                if let Some(ret_value) = value {
                    let ty = self.compile_expr(ret_value)?;
                    if ty != self.func.sig.ret {
                        return Err(Error::TypeMismatch(ret_value.location.clone()));
                    }
                } else {
                    if self.func.sig.ret != AsmPyType::Void {
                        return Err(Error::Other(
                            "missing value for return statement",
                            stmt.location.clone(),
                        ));
                    }
                    self.emit(Instruction::I32Const(0));
                }
                self.emit(Instruction::Br(self.blockdepth));
                Ok(())
            }
            pyast::StatementType::Expression { expression } => {
                let ty = self.compile_expr(expression)?;
                if ty != AsmPyType::Void {
                    return Err(Error::Other(
                        "result type of statement must be void",
                        stmt.location.clone(),
                    ));
                }
                Ok(())
            }
            pyast::StatementType::Assign { targets, value } => {
                let target = match &targets[..] {
                    [a] => a,
                    _ => return Err(Error::UnsupportedStatement(stmt.location.clone())),
                };
                let target = match target.node {
                    pyast::ExpressionType::Identifier { ref name } => name,
                    _ => return Err(Error::UnsupportedStatement(target.location.clone())),
                };
                if let Some((ty, varno)) = self.vars.get(target).cloned() {
                    let expr_ty = self.compile_expr(value)?;
                    if expr_ty != ty {
                        return Err(Error::TypeMismatch(value.location.clone()));
                    }
                    self.emit(Instruction::SetLocal(varno))
                } else {
                    let expr_ty = self.compile_expr(value)?;
                    let varno = self.varno;
                    self.varno += 1;
                    self.vardecls.push(elements::Local::new(
                        varno,
                        expr_ty.try_to_value_type(&value.location)?,
                    ));
                    self.vars.insert(target.to_string(), (expr_ty, varno));
                    self.emit(Instruction::SetLocal(varno));
                }
                Ok(())
            }
            _ => Err(Error::UnsupportedStatement(stmt.location.clone())),
        }
    }

    fn compile_expr(&mut self, expr: &pyast::Expression) -> Result<AsmPyType> {
        match &expr.node {
            pyast::ExpressionType::Binop { op, a, b } => {
                let a_ty = self.compile_expr(a)?;
                let b_ty = self.compile_expr(b)?;
                if a_ty != b_ty {
                    return Err(Error::TypeMismatch(expr.location.clone()));
                }
                let ty = match a_ty {
                    AsmPyType::Primitive(p) => p,
                    AsmPyType::Void => return Err(Error::UnexpectedVoid(expr.location.clone())),
                };
                use pyast::Operator::*;
                use Instruction::*;
                macro_rules! unsup {
                    () => {{
                        return Err(Error::UnsupportedStatement(expr.location.clone()));
                    }};
                }
                macro_rules! arith_op {
                    (
                        $(($op:ident, (
                            $i32_instr:expr,
                            $i64_instr:expr,
                            $u32_instr:expr,
                            $u64_instr:expr,
                            $f32_instr:expr,
                            $f64_instr:expr
                        )),)*
                        match a {
                            $($other:tt)*
                        }
                    ) => {
                        #[allow(unreachable_code)]
                        match op {
                            $($op => match ty {
                                Primitive::I32 => self.emit($i32_instr),
                                Primitive::I64 => self.emit($i64_instr),
                                Primitive::U32 => self.emit($u32_instr),
                                Primitive::U64 => self.emit($u64_instr),
                                Primitive::F32 => self.emit($f32_instr),
                                Primitive::F64 => self.emit($f64_instr),
                            })*
                            $($other)*
                        }
                    };
                }
                arith_op!(
                    (Add, (I32Add, I64Add, I32Add, I64Add, F32Add, F64Add)),
                    (Sub, (I32Sub, I64Sub, I32Sub, I64Sub, F32Sub, F64Sub)),
                    (Mult, (I32Mul, I64Mul, I32Mul, I64Mul, F32Mul, F64Mul)),
                    // todo: int / int -> float
                    (
                        Div,
                        (unsup!(), unsup!(), unsup!(), unsup!(), F32Div, F64Div)
                    ),
                    (BitAnd, (I32And, I64And, I32And, I64And, unsup!(), unsup!())),
                    (BitOr, (I32Or, I64Or, I32Or, I64Or, unsup!(), unsup!())),
                    (BitXor, (I32Xor, I64Xor, I32Xor, I64Xor, unsup!(), unsup!())),
                    (
                        RShift,
                        (I32ShrS, I64ShrS, I32ShrU, I64ShrU, unsup!(), unsup!())
                    ),
                    (LShift, (I32Shl, I64Shl, I32Shl, I64Shl, unsup!(), unsup!())),
                    (
                        FloorDiv,
                        (I32DivS, I64DivS, I32DivU, I64DivU, unsup!(), unsup!())
                    ),
                    match a {
                        _ => unsup!(),
                    }
                );
                Ok(AsmPyType::Primitive(ty))
            }
            pyast::ExpressionType::Number { value } => match value {
                pyast::Number::Integer { value } => match value.to_i32() {
                    Some(i) => {
                        self.emit(Instruction::I32Const(i));
                        Ok(AsmPyType::Primitive(Primitive::I32))
                    }
                    None => Err(Error::BadNumber(expr.location.clone())),
                },
                pyast::Number::Float { value } => match value.to_f32() {
                    Some(f) => {
                        self.emit(Instruction::F32Const(f.to_bits())); // i think?
                        Ok(AsmPyType::Primitive(Primitive::F32))
                    }
                    None => Err(Error::BadNumber(expr.location.clone())),
                },
                _ => Err(Error::BadNumber(expr.location.clone())),
            },
            pyast::ExpressionType::Identifier { name } => {
                if let Some((ty, local)) = self.vars.get(name).cloned() {
                    self.emit(Instruction::GetLocal(local));
                    Ok(ty)
                } else {
                    Err(Error::UnknownIdentifier(
                        name.clone(),
                        expr.location.clone(),
                    ))
                }
            }
            pyast::ExpressionType::Call { function, args, .. } => {
                let name = match &function.node {
                    pyast::ExpressionType::Identifier { name } => name,
                    _ => return Err(Error::UnsupportedStatement(function.location.clone())),
                };
                if let Some(val_ty) = primitive_from_str(name) {
                    use elements::Instruction::*;
                    use Primitive::*;
                    let arg = match args.as_slice() {
                        [a] => a,
                        _ => {
                            return Err(Error::Other(
                                "wrong number of arguments",
                                expr.location.clone(),
                            ))
                        }
                    };
                    if let pyast::ExpressionType::Number { ref value } = arg.node {
                        match (val_ty, value) {
                            (I64, pyast::Number::Integer { value }) => {
                                if let Some(n) = value.to_i64() {
                                    self.emit(I64Const(n));
                                    return Ok(AsmPyType::Primitive(I64));
                                }
                            }
                            (U32, pyast::Number::Integer { value }) => {
                                if let Some(n) = value.to_u32() {
                                    self.emit(I32Const(n as i32));
                                    return Ok(AsmPyType::Primitive(I64));
                                }
                            }
                            (U64, pyast::Number::Integer { value }) => {
                                if let Some(n) = value.to_u64() {
                                    self.emit(I64Const(n as i64));
                                    return Ok(AsmPyType::Primitive(I64));
                                }
                            }
                            (F64, pyast::Number::Float { value }) => {
                                self.emit(F64Const(value.to_bits()));
                                return Ok(AsmPyType::Primitive(F64));
                            }
                            _ => {}
                        }
                    }
                    let arg_ty = match self.compile_expr(arg)? {
                        AsmPyType::Primitive(ty) => ty,
                        AsmPyType::Void => {
                            return Err(Error::UnexpectedVoid(expr.location.clone()))
                        }
                    };
                    let instr = match (arg_ty, val_ty) {
                        (I32, I32)
                        | (I64, I64)
                        | (F32, F32)
                        | (F64, F64)
                        | (U32, U32)
                        | (U64, U64)
                        | (I32, U32)
                        | (U32, I32)
                        | (I64, U64)
                        | (U64, I64) => return Ok(AsmPyType::Primitive(val_ty)),
                        // signed integers
                        (I32, I64) => I64ExtendSI32,
                        (I64, I32) => I32WrapI64,
                        // mixed integers
                        (I32, U64) => I64ExtendSI32,
                        (I64, U32) => I32WrapI64,
                        (U32, I64) => I64ExtendSI32,
                        (U64, I32) => I32WrapI64,
                        // unsigned integers
                        (U32, U64) => I64ExtendUI32,
                        (U64, U32) => I32WrapI64,
                        // floats
                        (F32, F64) => F64PromoteF32,
                        (F64, F32) => F32DemoteF64,
                        // ints -> floats
                        (I32, F32) => F32ConvertSI32,
                        (I64, F32) => F32ConvertSI64,
                        (I32, F64) => F64ConvertSI32,
                        (I64, F64) => F64ConvertSI64,
                        (U32, F32) => F32ConvertUI32,
                        (U64, F32) => F32ConvertUI64,
                        (U32, F64) => F64ConvertUI32,
                        (U64, F64) => F64ConvertUI64,
                        // floats -> ints
                        (F32, I32) => I32TruncSF32,
                        (F32, I64) => I64TruncSF32,
                        (F64, I32) => I32TruncSF64,
                        (F64, I64) => I64TruncSF64,
                        (F32, U32) => I32TruncUF32,
                        (F32, U64) => I64TruncUF32,
                        (F64, U32) => I32TruncUF64,
                        (F64, U64) => I64TruncUF64,
                    };
                    self.emit(instr);
                    return Ok(AsmPyType::Primitive(val_ty));
                }
                let (fno, sig) = match self.compiler.items.get(name) {
                    Some(item) => item.as_func_sig().ok_or_else(|| {
                        Error::UnknownIdentifier(name.clone(), function.location.clone())
                    })?,
                    None => {
                        return Err(Error::UnknownIdentifier(
                            name.clone(),
                            function.location.clone(),
                        ))
                    }
                };
                if args.len() != sig.params.len() {
                    return Err(Error::Other(
                        "wrong number of arguments",
                        expr.location.clone(),
                    ));
                }
                for (ty, arg) in sig.params.iter().zip(args.iter()) {
                    let arg_ty = self.compile_expr(arg)?;
                    if &arg_ty != ty {
                        return Err(Error::TypeMismatch(arg.location.clone()));
                    }
                }
                self.emit(Instruction::Call(fno));
                Ok(sig.ret.clone())
            }
            _ => Err(Error::UnsupportedStatement(expr.location.clone())),
        }
    }

    fn finish(self) -> builder::FunctionDefinition {
        self.fbuilder
            .with_signature(sig_convert!(&self.func.sig))
            .body()
            .with_instructions(self.instructions)
            .with_locals(self.vardecls)
            .build()
            .build()
    }
}

struct AsmPyFunction {
    params: Vec<String>,
    sig: FuncTypeSig,
    body: pyast::Suite,
    export: bool,
    fno: u32,
}

enum AsmPyItem {
    Function(AsmPyFunction),
    TypeAlias(AsmPyType),
    ExternFunction { sig: FuncTypeSig, fno: u32 },
}

impl AsmPyItem {
    fn as_func_sig(&self) -> Option<(u32, &FuncTypeSig)> {
        match self {
            AsmPyItem::Function(AsmPyFunction { sig, fno, .. })
            | AsmPyItem::ExternFunction { sig, fno } => Some((*fno, sig)),
            _ => None,
        }
    }
}
