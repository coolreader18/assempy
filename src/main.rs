use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = pico_args::Arguments::from_env();
    let args = args.free_os()?;
    if args.len() != 2 {
        return Err("Wrong number of arguments".into());
    }
    let mut args = args.into_iter();
    let file = args.next().unwrap();
    let out_file = args.next().unwrap();
    let source = fs::read_to_string(file)?;
    let prog = rustpython_parser::parser::parse_program(&source)?;
    let module = assempy::compile(prog)?;
    parity_wasm::serialize_to_file(out_file, module)?;
    // // fs::write("out.wasm", &out)?;
    // let wabt = wabt::Wasm2Wat::new().convert(out)?;
    // let wabt = String::from_utf8_lossy(wabt.as_ref());
    // println!("{}", wabt);
    Ok(())
}
