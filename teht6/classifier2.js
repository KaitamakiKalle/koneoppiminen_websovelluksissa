const tf = require('@tensorflow/tfjs');
const knn = require('@tensorflow-models/knn-classifier');

/*
tehtävän versio jossa normalisoidaan kaikki ominaisuus arvot erikseen eikä samana taulukkona
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
  return numlabels;
}

// Erotetaan datasetistä ominaisuudet omaan taulukkoonsa
// separate features määrittää tehdäänkö ominaisuuksista omat taulukkonsa vai pidetäänkö ne yhdessä
function featuresFrom2Darr(arr, labelIndex, separatefeatures = false) {
  const featuresArr = arr.map((subarr) => {
    return subarr.filter((elem) => elem !== subarr[labelIndex]);
  });

  if (separatefeatures) {
    const arrs = [];
    for (let i = 0; i < featuresArr[0].length; i++) {
      let mem = [];
      featuresArr.forEach((elem) => {
        mem.push(elem[i]);
      });
      arrs.push(mem);
    }
    return arrs;
  } else {
    return featuresArr;
  }
}

// funktio normalisointia varten
function nlize(val, max, min) {
  return (val - min) / (max - min);
}

// funktio 2 ulotteisen taulukon normalisointiin
// featureArr on ominaisuus taulukko joss eri ominaisuuksia ei ole eroteltu
/*
sepFeatureArr on taulukko joka sisältää ominaisuudet omissa taulukoissaan.
Tätä tarvitaan jotta saadaan min ja max arvot normalisointia varten 
*/
function nlize2DfeatureArr(featuresArr, sepFeatureArr) {
  const nlizedArr = [];
  featuresArr.forEach((subarr) => {
    let mem = [];
    for (let i = 0; i < subarr.length; i++) {
      mem.push(
        nlize(
          subarr[i],
          Math.max(...sepFeatureArr[i]),
          Math.min(...sepFeatureArr[i])
        )
      );
    }
    nlizedArr.push(mem);
  });
  return nlizedArr;
}
// labelit eli ys arvot
const labels = labelsFrom2DArr(trainingData, 2);
// ominaisuudet eli xs arvot, ei normalisoitu
const features = featuresFrom2Darr(trainingData, 2, false);
// taulukko jossa ominaisuudet on eroteltu omiin taulukkoihinsa
// näistä taulukoista saadaan jokaiselle ominaisuudelle min/max arvot ja ne voidaan normalisoida erikseen
const featuresSep = featuresFrom2Darr(trainingData, 2, true);
// Normalisoidut xs arvot
const featuresNormlized = nlize2DfeatureArr(features, featuresSep);

const classifier = knn.create();
// classifierin harjoittaminen
for (let i = 0; i < trainingData.length; i++) {
  classifier.addExample(tf.tensor(featuresNormlized[i]), labels[i]);
}

const testFeatures = nlize2DfeatureArr([[300, 2]], featuresSep);

const prediction = classifier.predictClass(tf.tensor(testFeatures[0]), (k = 2));
prediction.then((response) => {
  console.log(response);
});
