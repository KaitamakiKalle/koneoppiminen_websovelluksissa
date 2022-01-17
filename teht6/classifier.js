const tf = require('@tensorflow/tfjs');
const knn = require('@tensorflow-models/knn-classifier');

/*
Tehtävän versio jossa ominaisuudet on normalisoitu yhdessä taulukossa keskenään
*/

// paino, pituus, label
const trainingData = [
  [303, 2, 'banaani'],
  [370, 1, 'omena'],
  [298, 2, 'banaani'],
  [277, 2, 'banaani'],
  [377, 3, 'omena'],
  [299, 2, 'banaani'],
  [382, 1, 'omena'],
  [374, 3, 'omena'],
  [303, 3, 'banaani'],
  [309, 2, 'banaani'],
  [301, 1, 'omena'],
  [366, 1, 'omena'],
  [311, 2, 'banaani'],
  [302, 2, 'banaani'],
  [313, 1, 'omena'],
  [373, 3, 'omena'],
  [305, 2, 'banaani'],
  [371, 2, 'omena'],
];

// muunnettaan labelit omaksi taulukokseen ja numeromuotoon
function labelsFrom2DArr(arr, labelIndex) {
  // labels taulukkoon kerätään yksi jokaista labelia
  const labels = [];

  // tehdään ensin taulukko joka sisältää kaikki labelit string muodossa
  const strLabels = arr.map((subarr) => {
    // pushataan jokaista labelia 1kpl labels taulukkoon
    // määritämme tämän labels taulukossa olevan labelin indeksin perusteella labelille numeroarvon
    if (labels.indexOf(subarr[labelIndex]) < 0) {
      labels.push(subarr[labelIndex]);
    }
    // mappi kerää labelit strLabels taulukkoon
    return subarr[labelIndex];
  });

  //Katsotaan labels taulukosta labelin indeksi josta tulee labelin numero arvo
  const numlabels = strLabels.map((elem) => {
    return labels.indexOf(elem);
  });
  console.log(labels);
  return numlabels;
}
// Erotetaan datasetistä ominaisuudet omaan taulukkoonsa
// separate features määrittää tehdäänkö ominaisuuksista omat taulukkonsa vai pidetäänkö ne yhdessä
function featuresFrom2Darr(arr, labelIndex) {
  const featuresArr = arr.map((subarr) => {
    return subarr.filter((elem) => elem !== subarr[labelIndex]);
  });

  return featuresArr;
}

function nlize(val, max, min) {
  return (val - min) / (max - min);
}
// normalisointi
function normalizeArr(arr, getminmax = false, maxval, minval) {
  let max;
  let min;
  if (maxval && minval) {
    // taulukon maksimi jos annettu parametrinä
    max = maxval;
    // taulukon minimi jos annettu parametrinä
    min = minval;
  } else {
    // tehdään 2d taulukosta 1 ulotteinen jotta min ja max arvot saadaan määritettyä
    const minmaxarr = arr.flat();
    // taulukon maksimi jos ei annettu parametrinä
    max = Math.max(...minmaxarr);
    // taulukon minimi jos ei annettu parametrinä
    min = Math.min(...minmaxarr);
  }

  // käydään jokainen elementti läpi
  const normlizedArr = arr.map((subarr) => {
    return subarr.map((elem) => {
      // jokainen elementti normalisoidaan nlize funktiolla
      return nlize(elem, max, min);
    });
  });
  // Jos min ja max arvot halutaan palauttaa myöhempää käyttöä varten palautetaan objekti
  // Objekti sisältää min ja max arvot sekä normalisoidun taulukon
  if (getminmax) {
    return {
      normlizedArr,
      max,
      min,
    };
  } else {
    // jos min ja max arvoja ei haluta palauttaa palautetaan vain normalisoitu taulukko
    return normlizedArr;
  }
}

// labelit
const labels = labelsFrom2DArr(trainingData, 2);
// ominaisuudet, ei normalisoitu

const features = featuresFrom2Darr(trainingData, 2, false);
// normalisoidut ominaisuudet
const {
  normlizedArr: featuresNormlized,
  max,
  min,
} = normalizeArr(features, true);

const classifier = knn.create();

// classifierin harjoittaminen
for (let i = 0; i < trainingData.length; i++) {
  classifier.addExample(tf.tensor(featuresNormlized[i]), labels[i]);
}

// myös ennustettavat arvot pitää normalisoida
const testFeatures = normalizeArr([[370, 3]], false, max, min);

const prediction = classifier.predictClass(tf.tensor(testFeatures[0]), (k = 2));
prediction.then((response) => {
  console.log(response);
});
