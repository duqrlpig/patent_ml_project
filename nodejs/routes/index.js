var express = require('express');
var router = express.Router();

var multer = require('multer');
var preprocessor = require('../module/preprocess')


router.get('/', function(req, res, next) {
  res.render('main.html');
});

//파일입력
router.post('/file', multer({ dest: '../../data/'}).single('patent'), async function(req,res){

  var csvdata = await preprocessor.readToCsv(req.file.path)
  res.render('result.html', {csvdata} );

});


module.exports = router;
