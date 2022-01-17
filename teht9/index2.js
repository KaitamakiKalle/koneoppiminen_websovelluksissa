// recognizer muuttujaan luodaan speechCommands malli
let recognizer;

/*
predictWord() kuuntelee ääntä ja kirjoittaa kuulemansa sanan selaimeen
kyseinen funktio vain havainnollistaa valmiin speechsCommands modelin toimintaa eikä sitä tarvita 
myöhemmin tehtävässä
*/
function predictWord() {
  // Array of words that the recognizer is trained to recognize.
  // words muuttujaan tallennetaan taulukko jossa ovat sanat joita model on opetettu tunnistamaan
  const words = recognizer.wordLabels();
  // console.log(words);
  // listen() metodin argumentiksi tulee callback funktio jota kutsutaan aina kun sana tunnistetaan
  recognizer.listen(
    // scores sisältää todennäköisyydet jotka vastaavat words taulukon sanoja
    ({ scores }) => {
      // Turn scores into a list of (score,word) pairs.
      // Scores muuttujaan tallennetaan millä todennäköisyydellä kuultu ääni on mikäkin sana
      scores = Array.from(scores).map((s, i) => ({ score: s, word: words[i] }));
      console.log(scores);
      // console.log(scores);
      // Find the most probable word.
      // Järjestetään taulukko niin että todennäköisin sana on ylimmäisenä
      scores.sort((s1, s2) => s2.score - s1.score);
      document.querySelector('#console').textContent = scores[0].word;
    },
    { probabilityThreshold: 0.75 }
  );
}

async function app() {
  // Luodaan malli muuttujaan ja valitaan audio inputin tyyppi
  recognizer = speechCommands.create('BROWSER_FFT');
  // ensureModelLoaded varmistaa että malli ja metadata latautuivat oikein
  await recognizer.ensureModelLoaded();
  // predictWord();
  buildModel();
}

app();
// One frame is ~23ms of audio.
// const NUM_FRAMES = 3;
// Muutetaan kerättyjen framejen määrää isommaksi koska haluamme kerätä nyt kokonaisia sanoja
const NUM_FRAMES = 23;
// taulukkoon tallennetaan kerätyt esimerkit
let examples = [];

// collect funktiota kutsutaan aina kun painetaan nappia left, right tai noise
// collect kerää ääninäytteitä nappia painettaessa
function collect(label) {
  if (recognizer.isListening()) {
    return recognizer.stopListening();
  }
  if (label == null) {
    return;
  }
  // äänen kerääminen valmiilla mallilla
  //---------------------------------------------------
  // Kuuntelu toteutetaan listen() metodilla
  recognizer.listen(
    async ({ spectrogram: { frameSize, data } }) => {
      // Data normalisoidaan jotta vältytään numeerisilta ongelmilta
      // Koska halusimme tutoriaalissa vain lyhyitä ääniä (esim. sormien napsautus) otetaan
      //       tässä vain viimeiset 3 framea huomioon eli ~ 70ms äänestä
      let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      // lisätään data esimerkki taulukkoon
      examples.push({ vals, label });
      document.querySelector(
        '#console'
      ).textContent = `${examples.length} examples collected`;
    },
    {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true,
    }
  );
  //---------------------------------------------------
}

// Normalisointi funktio
//---------------------------------------------------
function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map((x) => (x - mean) / std);
}
//---------------------------------------------------
// NUM_FRAMES kertoo audio framejen lukumäärän joista jokainen on 23ms pitkä
// jokaisessa framessa on 232 numeroa jotka merkitsevöt eri taajuuksia
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

// Uuden oman modelin harjoitus
//---------------------------------------------------
// Funktio jolla modeli harjoitetaan
async function train() {
  toggleButtons(false);
  // muutetaan labelit onhot muotoon
  const ys = tf.oneHot(
    examples.map((e) => e.label),
    3
  );

  const xsShape = [examples.length, ...INPUT_SHAPE];
  // luodaan xs tensori
  //console.log(examples);
  const xs = tf.tensor(flatten(examples.map((e) => e.vals)), xsShape);
  // xs.print();
  // modelin harjoitus
  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      // seurataan harjoituksen edistymistä tulostamalla tarkkuus ja epoch
      onEpochEnd: (epoch, logs) => {
        document.querySelector('#console').textContent = `Accuracy: ${(
          logs.acc * 100
        ).toFixed(1)}% Epoch: ${epoch + 1}`;
      },
    },
  });
  // dispose() metodi tuhoaa tensorit muistista kun niitä ei enää tarvita
  // näin vältetään ylikuormittamasta muistia
  tf.dispose([xs, ys]);
  toggleButtons(true);
}
//---------------------------------------------------

// Uuden oman modelin luonti
//---------------------------------------------------
// buildModel() rakentaa modelin jonka harjoitamme valmiista speechCommands modelista saadulla datalla
function buildModel() {
  model = tf.sequential();
  // audio dataa käsitellään convolutionaalisella layerilla koska kyseessä on ääninäyte
  // Jos ääninäyteitä analysoitaisiin tavallisella dense layerilla ennustus olisi liian tarkka
  // Ja äänen pitäisi olla aina täysin samanlainen
  model.add(
    tf.layers.depthwiseConv2d({
      depthMultiplier: 8,
      kernelSize: [NUM_FRAMES, 3],
      activation: 'relu',
      inputShape: INPUT_SHAPE,
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
  // flatten layer "litistää siihen tulevan syötteen 1d jolloin output on 2d"
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  const optimizer = tf.train.adam(0.01);
  // modelin kääntäminen
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
}
//---------------------------------------------------

// toggleButtons funktiolla napit otetaan pois käytöstä esim. harjoituksen ajaksi
// Vältytään virhetilanteilta jossa esim. harjoittaessa syötettäisiin lisää dataa harjoitusdatasettiin
function toggleButtons(enable) {
  document.querySelectorAll('button').forEach((b) => (b.disabled = !enable));
}

// Luodaan datasta 32 bittinen float point number taulukko josta tehdään myöhemmin input tensori
function flatten(tensors) {
  //console.log(tensors);
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  //console.log(result);
  return result;
}
// Funktio jolla slideria liikutetaan äänen perusteella
// Jos labelin arvo on 0 liikkuu slider vasemmalle ja jos se on 1 liikkuu slider oikealle
// Jos arvo on 2 = noise slider ei liiku
// async function printWord(labelTensor) {
//   const label = (await labelTensor.data())[0];
//   document.getElementById('console').textContent = label;
//   if (label == 2) {
//     return;
//   }
//   let delta = 0.1;
//   const prevValue = +document.getElementById('output').value;
//   document.getElementById('output').value =
//     prevValue + (label === 0 ? -delta : delta);
// }

async function printWord(labelTensor) {
  const label = (await labelTensor.data())[0];
  document.getElementById('console').textContent = label;
  if (label == 2) {
    return;
  }
  // tulostetaan moi tai hei sen mukaan kumpi sanotaan
  if (label === 0) {
    document.getElementById('console2').textContent = 'moi';
    setTimeout(() => {
      document.getElementById('console2').textContent = '';
    }, 1000);
  } else if (label === 1) {
    document.getElementById('console2').textContent = 'hei';
    setTimeout(() => {
      document.getElementById('console2').textContent = '';
    }, 1000);
  }
  //   let delta = 0.1;
  //   const prevValue = +document.getElementById('console2').value;
  //   document.getElementById('console2').value =
  //     prevValue + (label === 0 ? -delta : delta);
}

// listen() kuuntelee mikkiä ja tekee reaaliaikaisia ennustuksia
function listen() {
  if (recognizer.isListening()) {
    recognizer.stopListening();
    toggleButtons(true);
    document.getElementById('listen').textContent = 'Listen';
    return;
  }
  toggleButtons(false);
  document.getElementById('listen').textContent = 'Stop';
  document.getElementById('listen').disabled = false;

  recognizer.listen(
    async ({ spectrogram: { frameSize, data } }) => {
      // myös data jonka perusteella halutaan ennustaa pitää normalisoida
      const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
      // luodaan ennustus probs muuttujaan
      const probs = model.predict(input);
      // Ennustus on taulukko todennäköisyyksistä joista jokainen vastaa jotakin annetuista ääninäytteistä
      // argMax() metodilla poimitaan suurin todennäköisyys
      const predLabel = probs.argMax(1);
      await printWord(predLabel);
      // puhdistetaan turhat tensorit lopuksi muistista
      tf.dispose([input, probs, predLabel]);
    },
    {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true,
    }
  );
}
