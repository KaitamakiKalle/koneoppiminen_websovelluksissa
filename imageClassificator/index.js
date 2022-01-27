const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
// Funktio saa parametrina kansion osoitteen jossa kuvat ovat ja labelin joka kyseisille kuville annetaan
// Kuvat otetaan kansiosta ja muutetaan tensoreiksi jonka jälkeen niistä tehdään objekteja
// objetkit sisältävät kuva tensorin ja sitä vastaavan label arvon eli muoto on {xs, ys}
// Funktio toimii vain kun kyseessä on kuvien luokittelu koska parametrinä määritelty label annetaan kaikille
// kansiossa oleville kuville (esim. 0 tai 1)
async function imgsToObjArr(imgArrSrc = '', label) {
  // Objektit tallennetaan arr taulukkoon
  const arr = [];
  // Haetaan kaikki tiedostonimet readdirSync metodilla
  const filenames = await fs.readdirSync(imgArrSrc);

  // Jokainen tiedosto käydään erikseen lävitse
  await filenames.forEach((file) => {
    // tarkistetaan että tiedoston pääte on jpg tai jpeg jotta mahdolliset virheelliset tiedostot saadaa
    // karsittua pois
    if (path.extname(file) == '.jpg' || path.extname(file) == '.jpeg') {
      // koska kansion tiedostojen nimet ovat sring muodossa filenames taulukossa saadaan kansiopolusta ja
      // filenames taulukon alkiosta yksittäisen kuvan osoite readFileSync metodille
      const img = fs.readFileSync(`${imgArrSrc}/${file}`);

      // Kun kuva on haettu img muuttujaan se muutetaan tensoriksi tensorflown decodeImage metodilla
      const imgtensor = tf.node.decodeImage(img, 3);

      // Kuvat ovat yleensä valtavan kokoisia ja maskin tunnistamiseen kuvasta ei todellakaan tarvita
      // kovinkaan tarkkaa kuvaa joten kuvat pienennetään 64x64 pikselin kokoisiksi
      const imgResized = tf.image.resizeBilinear(imgtensor, [64, 64]);

      // lisätään kuva parametrina annetun labelin kanssa palautettavaan taulukkoon
      arr.push({ img: imgResized, label: label });
    }
  });

  return arr;
}

// Funktio erottaa xs ja ys arvot omiin taulukkoihinsa objekti taulukosta
function getValsFromObjArr(objArr) {
  // otetaan xs arvot mapilla omaan taulukkoonsa
  const xsArr = objArr.map((elem) => {
    return elem.img;
  });
  // otetaan ys arvot mapilla omaan taulukkoonsa ja muutetaan ne onehot muotoon
  const ysArr = objArr.map((elem) => {
    return elem.label === 1 ? [1, 0] : [0, 1];
    // return elem.label;
  });
  return {
    xsArr,
    ysArr,
  };
}

// Funktio normalisoi tensorin
// Funktiolle voidaan syöttää minimi ja maksimi arvot jolloin se käyttää niitä eikä laske arvoja
// syötetystä tensorista
// Jos arvoja ei anneta ne lasketaan tensorista ja myös palautetaan myöhempää käyttöä varten
async function normalize(tensor, maxVal, minVal) {
  // Jos min ja max arvot on annettu käytetään niitä
  if (maxVal && minVal) {
    const max = maxVal;
    const min = minVal;
    return tensor.sub(min).div(max.sub(min));
  } else {
    // muulloin ne lasketaan syötetystä tensorista
    const max = tf.max(tensor);
    const min = tf.min(tensor);
    // Tällöin palautuu objekti joka sisältää normalisoidun tensorin ja min sekä max arvot
    return { normlizedTensor: tensor.sub(min).div(max.sub(min)), min, max };
  }
}

// createModel luo modelin
// Koska kyseessä on kuvanluokittelu tehtävä käytetään convolutionaalista neuroverkkoa
/*
Verkon rakenne on:

convolutional layer(relu) + Maxpooling layer
convolutional layer(relu) + Maxpooling layer
convolutional layer(relu) + Maxpooling layer
flatten layer
dense layer(relu)
dense layer(relu)
dense layer(softmax) eli output

*/

// Funktio jolla voidaan erottaa taulukosta haluttu määrä testi dataa
// parametrina annetava taulukko josta data halutaan erottaa mutatoidaan tarkoituksella ettei
// testidata sekoitu harjoitusdataan
function separateTestData(amount, arr) {
  // Leikataan splicella taulukosta haluttu määrä testidataa
  const testArr = arr.splice(
    arr.length * (1 - amount),
    arr.length - arr.length * amount
  );
  return testArr;
}

function createModel() {
  const model = tf.sequential();
  // Convolutionaalinen layer
  model.add(
    tf.layers.conv2d({
      filters: 16,
      kernelSize: [3, 3],
      activation: 'relu',
      inputShape: [64, 64, 3],
    })
  );
  // pooling layer
  model.add(tf.layers.maxPooling2d([4, 4]));
  model.add(
    tf.layers.conv2d({
      filters: 8,
      kernelSize: [3, 3],
      activation: 'relu',
    })
  );

  model.add(tf.layers.maxPooling2d([2, 2]));
  model.add(
    tf.layers.conv2d({
      filters: 2,
      kernelSize: [3, 3],
      activation: 'relu',
    })
  );
  model.add(tf.layers.maxPooling2d([1, 1]));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 5, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

  // Modelin kääntäminen
  model.compile({
    optimizer: tf.train.adam(0.01),
    // Koska kyseessä on kuvanluokittelu jossa output on muodossa [0,1] eli sisältää kaksi todennäköisyyttä
    // käytetään categoricalCrossentropy loss funktiota
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

async function run() {
  // haetaan kuvat ja labelit

  // Maskittomien kuvat sijaitsevat './0' kansiossa ja maskillisten kuvat './1' kansiossa

  // Maskittomien kuvat
  const imagesObj0 = await imgsToObjArr('./0', 0);

  // Maskillisten kuvat
  const imagesObj1 = await imgsToObjArr('./1', 1);

  // yhdistetään taulukot myöhempää käsittelyä varten
  const imagesObjmerged = imagesObj0.concat(imagesObj1);

  // Koska data on tällä hetkellä järjestetty niin että maskittomien kuvat ovat peräkkäin ennen maskillisten kuvia
  // pitää data sekoittaa että neuroverkko saa kumpiakin kuvia harjoittaessa satunnaisessa järjestyksessä
  // Sekoitetaan tensorflown shuffle() metodilla
  tf.util.shuffle(imagesObjmerged);

  // Muutetaan {kuva, label} objektejen kuvat ja labelit omiksi taulukoikseen
  // Tuloksena on siis xs taulukko ja ys taulukko
  const { xsArr, ysArr } = await getValsFromObjArr(imagesObjmerged);

  // Kuinka paljon testidataa halutaan erottaa 0.1 = 10%
  const testAmount = 0.1;

  // Erotetaan splicella xs taulukosta haluttu määrä testidataa
  const testXsArr = separateTestData(testAmount, xsArr);
  // Erotetaan splicella ys taulukosta haluttu määrä testidataa
  const testYsArr = separateTestData(testAmount, ysArr);

  // Taulukot muunnetaan tensoreiksi

  // koska xsArr taulukko sisältää tensoreita muutetaan se tensoriksi käyttämällä tensorflown
  // stack() metodia. Metodi muuttaa juurikin tensoreita sisältävän taulukon tensoriksi
  const xsTensorUnNlized = tf.stack(xsArr);

  // xs tensori pitää vielä normalisoida
  // koska max ja min arvoja tarvitaan vielä myöhemmin ne otetaan myös talteen
  // muuttujiin max ja min
  const {
    normlizedTensor: xsTensor,
    max,
    min,
  } = await normalize(xsTensorUnNlized);

  // muutetaan ys taulukko tensoriksi
  const ysTensor = await tf.tensor(ysArr);

  // Koska haluamme käyttää min ja max arvoja myöhemmin valmiin tallennetun modelin kanssa
  // Tallennetaan ne tekstitiedostoon
  console.log(max);
  const maxData = `${max.dataSync()}`;
  const minData = `${min.dataSync()}`;
  fs.writeFileSync('./min_max_vals/max.txt', maxData);
  fs.writeFileSync('./min_max_vals/min.txt', minData);

  // Luodaan model
  const model = await createModel();

  // Harjoitetaan model harjoitusdatalla
  await model.fit(xsTensor, ysTensor, {
    epochs: 50,
    // Validationsplit ottaa annetun määrän harjoitusdataa joka kierroksella ja validoi kierroksen tulosta
    // jo harjoittaessa
    validationSplit: 0.1,
    callbacks: {
      // Seurataan lossia harjoituksen ajan console.log:lla  epoch ja loss
      onEpochEnd: async (epoch, logs) => {
        console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss);
      },
    },
  });

  // Tehdään testi xs arvoista tensori ja normalisoidaan sen arvot. Huomaa että tässä käytetään
  // jo tiedettyjä min ja max arvoja
  const testTensorXS = await normalize(tf.stack(testXsArr), max, min);

  // Tehdään test ys arvoista tensori
  const testTensorYS = tf.tensor(testYsArr);

  // Evaluoidaan modeli testidatalla
  const result = await model.evaluate(testTensorXS, testTensorYS, {
    batchSize: 32,
  });

  console.log('loss');
  // evaluoinnin loss
  result[0].print();
  console.log('accuracy');
  // evaluoinnin accuracy
  result[1].print();

  // Testataan modelia vielä omalla oikealla kuvalla
  // haetaan kuva readFileSync metodilla
  const testImage = await fs.readFileSync('./testImages/test1.jpg');
  // Muutetaan kuva tensoriksi decodeImage metodilla
  let testImgTensor = await tf.node.decodeImage(testImage, 3);
  // Muutetaan kuva oikeaan kokoon
  testImgTensor = await tf.image.resizeBilinear(testImgTensor, [64, 64]);
  // Normalisoidaan tensori
  testImgTensor.print();
  testImgTensor = await normalize(testImgTensor, max, min);

  // Laajennetaan testi tensorin ulottovuutta jotta muoto on sopiva inputille
  testImgTensor = testImgTensor.expandDims(0);

  // suoritetaan ennustus
  const prediction = model.predict(testImgTensor);
  // tulostetaan ennustus
  prediction.print();
  // ennustuksen ensimmäinen alkio kertoo millä todennäköisyydellä kuvan henkilöllä on maski naamalla
  console.log(prediction.dataSync()[0]);

  // Tallennetaan modeli
  // Kommentoitu pois koska hyvä modeli on jo tallennettuna
  // const saveResult = await model.save(
  //   'file:////Users/kallekaitamaki/Documents/imageClassificator/faceMaskClassifier'
  // );
}

run();
