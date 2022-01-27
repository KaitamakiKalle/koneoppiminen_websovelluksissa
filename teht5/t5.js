const tf = require('@tensorflow/tfjs-node');

// csv datan tiedostopolku
const csvPath = 'file://t5data.csv';

async function printobj(declaration, obj, iterations) {
  const it = await obj.iterator();
  console.log(declaration);
  for (let i = 0; i < iterations; i++) {
    const elem = await it.next();
    console.log(elem);
  }
  console.log('');
}

async function csvToObj(csvdata) {
  const dataObj = await tf.data.csv(csvPath, {
    columnConfigs: { laatuluokka: { isLabel: true } },
  });

  return dataObj;
}
async function objToArrObj(dataObj) {
  const dataToArrObj = await dataObj.map(({ xs, ys }) => {
    const labels = [
      ys.laatuluokka === 'A' ? 1 : 0,
      ys.laatuluokka === 'B' ? 1 : 0,
      ys.laatuluokka === 'C' ? 1 : 0,
    ];
    return { xs: Object.values(xs), ys: Object.values(labels) };
  });

  return dataToArrObj;
}

async function normalize(tensor) {
  const min = tf.min(tensor);
  const max = tf.max(tensor);

  return tensor.sub(min).div(max.sub(min));
}
async function dataToTensor(dataObj) {
  const it = await dataObj.iterator();
  const elem = await it.next();

  tensor = [elem.value.xs, elem.value.ys];
  return tensor;
}

async function run() {
  const dataObj = await csvToObj(csvPath);
  printobj('Data objektissa', dataObj, 1);
  const arrObj = await objToArrObj(dataObj);
  printobj('Data taulukoina objektissa', arrObj, 1);
  const tensorObj = await arrObj.batch(9);

  const tensor = await dataToTensor(tensorObj);
  //nomralisointi
  tensor[0] = await normalize(tensor[0]);

  tf.print(tensor);
  return tensor;
}

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [4], units: 50, activation: 'relu' }));
model.add(tf.layers.dense({ units: 30, activation: 'tanh' }));
model.add(tf.layers.dense({ units: 20, activation: 'relu' }));
model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

async function trainModel(model) {
  const data = await run();
  xs = data[0];
  ys = data[1];
  console.log(xs.shape);
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(0.06),
  });

  model.fit(xs, ys, { batchSize: 3, epochs: 100 });
}

trainModel(model);
