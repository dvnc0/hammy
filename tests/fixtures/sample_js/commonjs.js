var legacyHelper = function(x) {
    return x * 2;
};

const withFuncExpr = function compute(x) {
    return x;
};

function plainFunction(a, b) {
    return a + b;
}

module.exports = function handler(req, res) {
    res.send('ok');
};

exports.namedExport = (a, b) => {
    return a + b;
};
