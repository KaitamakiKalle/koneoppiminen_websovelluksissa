import * as funcs from './functions.js';
import tf from '@tensorflow/tfjs';

// Parsii datasta vain sen osan jota haluamme käyttää
async function dataParsing() {
  const rawdata = await funcs.getLocalData('./data/owid-covid-data.json');
  // rawdata on objekti joka sisältää kaikkien maiden tiedot
  // Objektin avaimet ovat 3 kirjaimisia maakoodeja joten suomen tiedot saadaan FIN avaimella
  const finlandData = rawdata.FIN;
  // mapataan data joka halutaan sisällyttää datasettiin
  const data = await finlandData.data.map((elem) => {
    return {
      // päivämääräs
      date: elem.date,
      // uudet tartunnat/päivä
      new_cases: elem.new_cases,
      // kaikki tartunnat
      total_cases: elem.total_cases,
      // kaikki kuolemat
      total_deaths: !elem.total_deaths ? 0 : elem.total_deaths,
      // uudet testit/päivä
      new_tests: !elem.new_tests ? 0 : elem.new_tests,
      // uudet rokotukset/päivä
      new_vaccinations: !elem.new_vaccinations_smoothed
        ? 0
        : elem.new_vaccinations_smoothed,
    };
  });
  // Viimeiset objektit sisältävät korruptoitunutta dataa joten leikataan ne pois
  return data.slice(0, data.length - 10);
}

// Funktio muuttaa datan taulukko muotoon
function mainNnData(data) {
  /* Datan muunnos muotoon [{
    xs:[],
    ys:[]
    
  }*/
  const dataToArrObj = data.map((elem) => {
    return {
      xs: [elem.new_cases, elem.new_tests, elem.new_vaccinations],
      ys: [elem.total_cases],
    };
  });
  // mapataan xs arvot omaan taulukkoonsa. tuloksena 2 ulotteinen taulukko
  const xs = dataToArrObj.map((elem) => {
    return elem.xs;
  });
  // mapataan ys arvot omaan taulukkoonsa. tuloksena 2 ulotteinen taulukko
  const ys = dataToArrObj.map((elem) => {
    return elem.ys;
  });

  // Normalisoidaan xs arvot nlize2dArr funktiolla.
  // Funktio palauttaa normalisoidun xs taulukon = xsnlized ja minimi/maksimi arvot myöhempää normalisointia varten
  const { nlizedArr: xsnlized, minMaxVals } = funcs.nlize2DArr(xs, true);

  return {
    xs: xsnlized,
    ys: ys,
    minMaxVals: minMaxVals,
  };
}

// modelin luonti
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [3] }));
model.add(tf.layers.dense({ units: 20, activation: 'relu' }));
model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1 }));

async function run() {
  // haetaan haluttu data datasetistä
  const dataParsed = await dataParsing();
  tf.util.shuffle(dataParsed);
  // mainNnData palauttaa objektin jossa on xs ja ys arvot sekä min ja max arvot myöhempää normalisointia varten
  const data = await mainNnData(dataParsed);

  // erotetaan xs ja ys sekä min max arvot
  const dataArr = [data.xs, data.ys];
  const minMaxVals = data.minMaxVals;

  // jaetaan data harjoitus ja testi dataan
  const trainingData = [
    // data[0] = xs arvo
    dataArr[0].slice(0, dataArr[0].length - 50),
    // data[1] = ys arvo
    dataArr[1].slice(0, dataArr[1].length - 50),
  ];

  const testData = [
    // data[0] = xs arvo
    dataArr[0].slice(dataArr[0].length - 49, dataArr[0].length - 1),
    // data[0] = xs arvo
    dataArr[1].slice(dataArr[1].length - 49, dataArr[1].length - 1),
  ];

  // muunnetaan harjoitusdata tensoreiksi
  const xs = tf.tensor(trainingData[0]);
  const ys = tf.tensor(trainingData[1]);

  // muunnetaan testidata tensoreiksi
  const testxs = tf.tensor(testData[0]);
  const testys = tf.tensor(testData[1]);

  // Modelin kääntäminen
  await model.compile({
    // loss funktiona käytetään meanSquaredErroria koska kyseessä on regressio
    loss: 'meanSquaredError',
    // optimoijana adam
    optimizer: tf.train.adam(0.1),
    // metriikoista kerätään accuracy jotta se voidaan tulostaa myöhemmin
    metrics: ['accuracy' /*, 'mse'*/],
  });

  // earlystopper harjoituksen pysäyttämiseksi kun loss ei enää pienene
  const earlyStopper = await tf.callbacks.earlyStopping({
    // monitoroidaan loss:a
    monitor: 'val_loss',
    // minDelta = pienin lossin muutos joka lasketaan parannukseksi
    minDelta: 0.1,
    // patience = kuinka monen epochin jälkeen harjoitus lopetetaan kun loss ei enää muutu
    patience: 15,

    mode: 'auto',
  });

  // Modelin harjoitus
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
    // modelia voidaan validoida jo harjoittaessa
    // validationSplit ottaa jokaisen epochin lopussa osan harjoitusdatasta ja validoi harjoitusta sen avulla
    validationSplit: 0.1,
    callbacks: {
      earlyStopper,
      // Seurataan lossia harjoituksen ajan console.log:lla  epoch ja loss
      onEpochEnd: async (epoch, logs) => {
        console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss);
      },
    },
  });
  // modelin evaluointi
  const result = await model.evaluate(testxs, testys, {
    batchSize: 32,
  });

  console.log('loss');
  // evaluoinnin loss
  result[0].print();
  // evaluoinnin accuracy
  result[1].print();
}
run();
