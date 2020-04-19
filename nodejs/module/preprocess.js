var fs = require('fs');

module.exports = {
    readToCsv: async (path) => {

        // 동기방식의 파일읽기. 파일을 읽은 후 data 변수에 저장
        var data = fs.readFileSync(path, 'utf-8');

        var num = 0;
        var csvdata = "";

        //<CLAIM ORDER='0'>
        var sidx = data.indexOf("<CLAIM ORDER=")
        var eidx = data.indexOf("</CLAIM>", sidx)

        while ( true ){
            sidx = data.indexOf("<CLAIM ORDER=", eidx)
            eidx = data.indexOf("</CLAIM>", sidx)

            if(sidx < 0 ) break;

            num += 1;
            csvdata += `${num},${data.substr(sidx + 2*num.toString().length + 19, eidx - sidx - 2*num.toString().length - 20)}\r\n`;
        }
        return csvdata;
    }
}
