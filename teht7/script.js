//--------------------------------

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
// Funktio datan hakemiseksi
async function getData() {
  // Data haetaan käyttämällä fetchiä
  const carsDataResponse = await fetch(
    'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
  );
  /* 
  Fetch palauttaa json muotoisen datan joka muutetaan 
  Objektiksi
  */
  const carsData = await carsDataResponse.json();
  //console.log(carsData);
  /* 
  Datatasta poimitaan vaan ne tiedot joita haluamme 
  käyttää tässä tapauksessa
  */
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    // filteroidaan data pois joissa valittuja ominaisuuksia ei ole
    .filter((car) => car.mpg != null && car.horsepower != null);

  return cleaned;
}

//--------------------------------

//--------------------------------

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  // Data laitetaan taulukkoon objekteina
  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));
  // Renderöidään selaimeen koordinaatisto jossa datapisteet näkyvät
  // Datan visualisoinnilla voidaan nähdä yhtäläisyyksiä datassa ja huomata virheellistä dataa
  // Myös normalisoinnin tarve voidaan havaita visualisoimalla dataa
  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300,
    }
  );
  // Create the model
  // model tallennetaan model muuttujaan
  const model = createModel();
  // Alla oleva tulostaa modelin summaryn (tiedot) selaimeen
  tfvis.show.modelSummary({ name: 'Model Summary' }, model);

  // Convert the data to a form we can use for training.
  // Kutsutaan convertTotensor funktioita ja tallenetaan sen palauttama objekti tensorData muuttujaan
  const tensorData = convertToTensor(data);
  // erotetaan vielä input ja label tensorit erilleen
  const { inputs, labels } = tensorData;

  // Train the model
  // kutsutaan traniModel funktiota joka harjoittaa modelin
  await trainModel(model, inputs, labels);
  console.log('Done Training');
  /*
Mallin harjoitus ei aina onnistu samalla tavalla.
Tämä johtuu muunmuassa siitä että sekoitamme datan shuffle() metodilla ja se voi joskus sekoittua parempaan 
järjestykseen harjoituksen kannalta ja tuottaa paremman tuloksen. Myös optimointi funktio saatta säätää
joka kerta painoarvoja hiukan eri tavalla. 
*/
  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
  // More code will be added below
}

//--------------------------------

//--------------------------------

// Funktio koneoppimismallin (neuroverkon) luomiseksi
function createModel() {
  // Create a sequential model
  // Luodaan tensorflow:n sequential malli
  const model = tf.sequential();

  // Add a single input layer
  /*
  Lisätään malliin yksi kerros jossa on yksi neuroni ja johon syötetään tensori jonka muoto on [1]
  kyseinen kerros toimii siis input kerroksena
  usebias: true tarkoittaa että kerroksessa käytetään bias arvoa joka lisätään paino arvoon tai arvoihin

  */
  model.add(
    tf.layers.dense({
      inputShape: [1],
      units: 1,
      activation: 'tanh',
      useBias: true,
    })
  );

  // Add an output layer
  /*
  Lisätään output kerros josta ennuste saadaan ulos. Kerros sisältää yhden neuronin
   */
  model.add(
    tf.layers.dense({ units: 1, activation: 'linear ', useBias: true })
  );

  return model;
}

//--------------------------------

//--------------------------------

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
// Funktio datan muuttamiseksi tensoreiksi
// Data muutetaan tensorimuotoon koska neuroverkkoon syötetään sisälle tensoreita
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  // tensorflow:n tidy() metodi tuhoaa sen sisällä käytetyt tensorit surituksen jälkeen lukuunottamatta
  // niitä jotka sieltä palautetaan
  return tf.tidy(() => {
    // Step 1. Shuffle the data
    // tf.util.shuffle() sekoittaa datan satunnaiseen järjestykseen
    // Tämä on usein hyödyllistä ja parantaa neuroverkon ennustuksen tarkkuutta
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    // luodaan inputs taulukko josta myöhemmin tulee neuroverkkoon sisään laitettava tensori
    const inputs = data.map((d) => d.horsepower);
    // luodaan labels taulukko jotka ovat tunnettuja vastauksia ja joita käytetään neuroverkkoa opettaessa
    const labels = data.map((d) => d.mpg);

    // muunnetaan taulukot 2 ulotteisiksi tensoreiksi
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    // Data normalisoidaan välille 0-1 min max normalisoinnilla
    // näin data skaalautuu koordinaatistolle paremmin ja saamme parempia tuloksia
    // Datan normalisointia varten tensoreista määritetään minimi ja maximi arvot
    // Koska laskeminen tapahtuu kaavalla (value-min)/(max-min)
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    // normalisointi tapahtuu tässä hyödyntäen tensorflow:n metodeita
    // sub() vähentää, div() jakaa
    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    // palautetaan tiedot objektissa
    // maximi ja minimi arvot palautetaan koska niitä tarvitaan myöhemmin
    // koska myös valmiiseen neuroverkkoon syötettävä data on normalisoitava
    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

//--------------------------------

//--------------------------------

// Funktio jolla modeli harjoitetaan
async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  // model.compile on tehtävä ennen harjoitusta.
  // model.compile määrittää optimointi funktion, loss funktion ja metrics evaluoitavat tiedot kuten 'accuracy'
  model.compile({
    // optimointi funktio päivittää neuroverkon painoarvoja kun sitä harjoitetaan
    optimizer: tf.train.adam(0.08),
    // loss kertoo kuinka tarkka harjoituksen tulos on kunkin harjoituskierroksen jälkeen
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  // batchSize määrittää kuinka monta datasetin 'alaryhmää' , subsettiä, modeli näkee joka iteraatiokierroksella
  const batchSize = 32;
  // Epochs kertoo monta kertaa model käy koko datasetin läpi
  const epochs = 100;

  // model.fit() aloittaa varsinaisen harjoituksen
  // fit() ottaa parametreinä input tensorin, output tensorin ja valinnaiset muut määrittelyt kuten
  // epochs ja batchsize
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    // Harjoituksen jälkeen tulostetaan graafit harjoituksen onnistumista
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    ),
  });
}

//--------------------------------

// Funktio jolla modeli evaluoidaan
function testModel(model, inputData, normalizationData) {
  // Erotetaan maximi ja minimi arvot objektista johon ne tallennettiin aikaisemmin
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    // linspace palauttaa tensorin jossa on tasaisin välein annettu määrä numeroita annetulta väliltä
    // tässä tapauksessa väliltä 0 ja 1 generoidaan 100 numeroa
    // näistä saadaan uudet 'esimerkit' modelille
    const xs = tf.linspace(0, 1, 100);

    // Ennustetaan modelilla x arvot ja tallennetaan ne muuttujaan preds
    const preds = model.predict(xs.reshape([100, 1]));

    // "poistetaan" normalisointi xs arvoista
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    // "poistetaan" normalisointi ennustetuista arvoista
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    //dataSync metodilla tensoreista saadaan typedArray joten dataa voidaan käsitellä tavallisella js:llä
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  // Muodostetaan taulukko jossa on objekteja tensorista
  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  // muodostetaan taulukko jossa on objekteja alkuperäisistä arvoista
  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  // Renderöidään koordinaatisto jolla verrataan alkuperäistä dataa ja evaluointidataa
  // näin nähdään helposti kuinka tarkka neuroverkko on
  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    {
      values: [originalPoints, predictedPoints],
      series: ['original', 'predicted'],
    },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300,
    }
  );
}
document.addEventListener('DOMContentLoaded', run);
