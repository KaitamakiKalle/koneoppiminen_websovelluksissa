const regression = require('regression');

/*
Tunnettua dataa joka sisältää krokotiilin ikävuosia ja niitä vastaavia pituuksia.
*/
const knownX = [1, 2, 3, 4, 5, 6]; // ikä v
const knownY = [1.0, 1.6, 2.0, 2.3, 2.6, 2.8]; // pituus m

function predictLength([...knownx], [...knowny], predictable) {
  const arr = knownx.map((x) => [x, knowny.shift()]);

  // for (let i = 0; i < knownx.length; i++) {
  //   arr.push([knownx[i], knowny[i]]);
  // }

  const regressionLog = regression.logarithmic(arr);

  return regressionLog.predict(predictable);
}

console.log(predictLength(knownX, knownY, 20));
console.log(predictLength(knownX, knownY, 50));
console.log(predictLength(knownX, knownY, 100));
