const jsregression = require('js-regression'); // js-regression -kirjastomoduuli

/*
TrainingData on on tunnettua (hatusta vedettyä) dataa, joka kertoo, onko henkilö [ikä, mitattu kuntotaso (1-10),
mitattu taitotaso (1-10), onnistunut suorittamaan tehtävän x (0==ei, 1==kyllä)].
*/
const trainingData = [
  [18, 1, 5, 0],
  [20, 3, 2, 0],
  [24, 1, 3, 0],
  [30, 8, 5, 1],
  [32, 7, 5, 0],
  [40, 9, 5, 1],
  [45, 5, 9, 1],
  [51, 8, 6, 0],
  [60, 4, 9, 1],
  [65, 2, 1, 0],
];

function logReg(a, l, i) {
  // regressio-olio logistic syntyy kirjaston algoritmilla
  const logistic = new jsregression.LogisticRegression({
    alpha: a,
    lambda: l,
    iterations: i,
  });

  /*** Harjoitetaan logistista regressiota harjoitusdatalla ***/
  const model = logistic.fit(trainingData);
  return model; // palautetaan harjoitettu malli
}
// Prediction saa parametreikseen harjoitetun mallin ja selittävän muuttujan
function prediction(model, x1, x2, x3) {
  // console.log(model);
  /*** Harjoitetusta mallista saadaan vakio ja kulmakerroin ***/
  const a = model.theta[0]; // vakio
  const b1 = model.theta[1]; // 1. kulmakerroin
  const b2 = model.theta[2]; // 2. kulmakerroin
  const b3 = model.theta[3]; // 3. kulmakerroin
  /*** Lasketaan todennäköisyys logistisen regression kaavalla ***/
  const probability = 1 / (1 + Math.exp(-(a + b1 * x1 + b2 * x2 + b3 * x3)));
  return probability;
}

const age = 25;
const kuntoTaso = 8;
const taitoTaso = 8;
const model = logReg(0.001, 0, 10000);
const proba = prediction(model, age, kuntoTaso, taitoTaso);

console.log(
  'Todennäköisyys sille että ' +
    age +
    ' vuoden ikäinen henkilö jonka kuntotaso on ' +
    kuntoTaso +
    ' ja taitotaso on ' +
    taitoTaso +
    ' onnistuu tehtävässä on ' +
    proba.toFixed(2)
);
