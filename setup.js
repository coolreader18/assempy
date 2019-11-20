const fs = require("fs");

WebAssembly.instantiate(fs.readFileSync("test.wasm"), {
  console: {
    log: arg => {
      console.log(arg);
      return arg;
    }
  }
})
  .then(({ instance }) => {
    global.mod = instance.exports;
  })
  .catch(err => {
    console.log(err);
    process.exit(1);
  });
