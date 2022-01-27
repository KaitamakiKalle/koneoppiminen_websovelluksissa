//-------------------------------------------------------------
// Tekemäni datankäsittely funkiot
// Normaalisti importaisin nämä omasta filestään mutta koska en tehtävää
// viitsinyt ruveta buildaamaan niin olen ne sisältänyt tähän samaan fileen
function nlize(val, max, min) {
  return (val - min) / (max - min);
}
function sepFeatFrom2Darr(arr) {
  const arrs = [];
  for (let i = 0; i < arr[0].length; i++) {
    let mem = [];
    arr.forEach((elem) => {
      mem.push(elem[i]);
    });
    arrs.push(mem);
  }

  return arrs;
}
/*
Funktion toimintaperiaate on se että se erottelee ensin eri ominaisuudet omiin
Taulukkoihinsa 2d taulukosta. Näistä taulukoista saadaan min ja max arvot normalisointia varten.
Koska tiedämme min ja max arvot jokaiselle ominaisuudelle voidaan taulukko normalisoida yksinkertaisella
tuplasilmukalla
*/
function nlize2DArr(featuresArr = [], returnMinMax = false, minMaxArr = []) {
  // Normalisoitu taulukko tallennetaan muuttujaan nlizedArr
  console.log(minMaxArr);
  try {
    const nlizedArr = [];
    // minMaxVals taulukkoon tallennetaan minimi ja maksimi arvot normalisointia varten
    let minMaxVals = [];

    // Jos minMaxArr syötetään parametrinä arvot saadaan siitä suoraan
    if (minMaxArr.length > 0 && Array.isArray(minMaxArr)) {
      minMaxArr.forEach((subarr) => {
        if (!Array.isArray(subarr) || subarr.some(isNaN)) {
          throw new Error('Array has some elements that are not numbers');
        }
      });

      minMaxVals = minMaxArr;
      // Muussa tapauksessa arvot lasketaan syötetystä taulukosta
      // Alitaulukoiden alkioista, ominaisuuksista kasataan omat taulukkonsa josta saadaan min ja max arvot
    } else if (minMaxArr.length <= 0) {
      console.log(minMaxArr);
      // sepFeatFrom2Darr kasaa ominaisuuksista omat taulukkonsa
      const sepFeatureArr = sepFeatFrom2Darr(featuresArr);
      console.log(sepFeatureArr);

      minMaxVals = sepFeatureArr.map((subarr) => {
        return [Math.max(...subarr), Math.min(...subarr)];
      });
    }

    featuresArr.forEach((subarr) => {
      let mem = [];
      for (let i = 0; i < subarr.length; i++) {
        mem.push(
          nlize(
            subarr[i],

            minMaxVals[i][0],
            minMaxVals[i][1]
          )
        );
      }
      nlizedArr.push(mem);
    });
    // Jos minimi ja maksimi arvot halutaan myöhempää käyttöä varten palautetaan myös ne_---
    if (returnMinMax) {
      return {
        nlizedArr,
        minMaxVals,
      };
    } else if (!returnMinMax) {
      return nlizedArr;
    }
  } catch (error) {
    console.error(error);
  }
}

//-------------------------------------------------------------

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

async function dataToTensor() {
  const xsunnlized = trainingData.map((subarr) => subarr.slice(0, 3));
  // Muutetaan labelit onehot muotoon samalla kun ne erotetaan omaksi taulukokseen
  // ys arvot ovat valmiiksi 0 ja 1 välillä joten sitä ei tarvitse normalisoida
  const ysunnlized = trainingData.map((subarr) =>
    subarr[3] === 0 ? [1, subarr[3]] : [0, subarr[3]]
  );
  // Normalisoidaan xs arvot
  // koska funktio palauttaa objektin joka sisältää myös maksimi ja minimi
  // arvot myöhempää normalisointia varten napataan nämä kaksi eri muuttujiin
  const { nlizedArr: nlizedxs, minMaxVals } = await nlize2DArr(
    xsunnlized,
    true
  );

  return {
    xs: tf.tensor(nlizedxs),
    ys: tf.tensor(ysunnlized),
    minMaxVals: minMaxVals,
  };
}

async function createModel(featureCount) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 12,
      activation: 'relu',
      inputShape: [featureCount],
    })
  );

  model.add(
    tf.layers.dense({
      units: 2,
      activation: 'softmax',
    })
  );
  return model;
}

async function run() {
  const data = await dataToTensor();
  // Normalisoidaan xs data välille 0-1

  //   const xsUnNorm = data.xs;
  //   const min = tf.min(xsUnNorm);
  //   const max = tf.max(xsUnNorm);
  //   const xs = await normalize(xsUnNorm, min, max);
  const xs = await data.xs;
  // Ys dataa ei tarvitse normalisoida koska se on jo valmiiksi pelkkiä 0 ja 1
  xs.print();
  const ys = await data.ys;
  const minMaxVals = data.minMaxVals;
  const model = await createModel(3);
  // learning ratea suurentamalla 0.001:stä 0.01:n tulos muuttui paremmaksi
  const optimizer = tf.train.adam(0.001);
  await model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  await model.fit(xs, ys, {
    epochs: 300,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss);
      },
    },
  });
  //   const test1 = await normalize(tf.tensor([[50, 1, 1]]), min, max);
  //   test1.print();
  //   const test2 = await normalize(tf.tensor([[50, 9, 9]]), min, max);
  const test1 = await tf.tensor(nlize2DArr([[50, 1, 1]], false, minMaxVals));
  const test2 = await tf.tensor(nlize2DArr([[50, 9, 9]], false, minMaxVals));
  const pred1 = await model.predict(test1);
  const result1 = await pred1.dataSync();
  const pred2 = await model.predict(test2);
  const result2 = await pred2.dataSync();
  console.log(result1);
  console.log(result2);

  document.write(
    'Todennäköisyys että henkilö 1 onnistuu tehtävässä: ' +
      result1[1] +
      '\n' +
      'Todennäköisyys että henkilö 2 onnistuu tehtävässä: ' +
      result2[1]
  );
}
run();
