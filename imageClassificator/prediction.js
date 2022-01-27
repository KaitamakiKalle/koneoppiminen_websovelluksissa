const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

function normalize(tensor, maxVal, minVal) {
  if (maxVal && minVal) {
    const max = maxVal;
    const min = minVal;
    return tensor.sub(min).div(max.sub(min));
  } else {
    const max = tf.max(tensor);
    const min = tf.min(tensor);
    return { normlizedTensor: tensor.sub(min).div(max.sub(min)), min, max };
  }
}
async function run() {
  const model = await tf.loadLayersModel(
    'file:///Users/kallekaitamaki/Documents/imageClassificator/faceMaskClassifier/model.json'
  );

  // Otetaan maksimi arvo txt tiedostosta normalisointia varten
  const max = tf.tensor(
    parseInt(fs.readFileSync('./min_max_vals/max.txt').toString())
  );

  // Otetaan minimi arvo txt tiedostosta normalisointia varten
  const min = tf.tensor(
    parseInt(fs.readFileSync('./min_max_vals/min.txt').toString())
  );

  // Haetaan kuva joka halutaan luokitella
  const testImage = fs.readFileSync('./testImages/test3.jpg');
  // Muutetaan kuva tensoriksi
  let testImgTensor = tf.node.decodeImage(testImage, 3);
  // Muutetaan kuva oikeaan kokoon
  testImgTensor = tf.image.resizeBilinear(testImgTensor, [64, 64]);
  // Normalisoidaan tensori
  testImgTensor = await normalize(testImgTensor, max, min);
  // Laajennetaan tensorin ulottuvuutta jotta se on sopivan muotinen inputille
  testImgTensor = testImgTensor.expandDims(0);

  // ennustetaan onko heknilöllä  maski naamalla
  const prediction = model.predict(testImgTensor);
  prediction.print();
  console.log('\n');
  console.log(
    'Kuvan henkilöllä on maski ' +
      Math.floor(prediction.dataSync()[0] * 100) +
      '% todennäköisyydellä'
  );
  console.log('\n');
}
run();
