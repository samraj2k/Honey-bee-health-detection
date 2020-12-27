const express = require("express");
const app = express();
const path = require("path");

const multer = require("multer");

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "./uploads");
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
  },
});

const upload = multer({ storage: storage });

app.set("view engine", "ejs");

app.post("/submit", upload.single("image"), async (req, res) => {
  console.log(req.file);

  let msg;

  const spawn = require("child_process").spawnSync;
  const pythonProcess = spawn("python", ["./scrip.py", req.file.filename]);

  if (!pythonProcess.status) {
    msg = { success: true, data: pythonProcess.stdout.toString() };
  } else {
    msg = { success: false, data: pythonProcess.stderr.toString() };
  }
  res.json(msg);
});

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname + "/views/home.html"));
});

app.get("/submit", (req, res) => {
  res.redirect("/");
});

app.listen(process.env.PORT || 5000, () => {
  console.log(`Listening on port ${process.env.PORT || 5000}`);
});
