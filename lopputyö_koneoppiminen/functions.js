import fetch from 'node-fetch';
import fs from 'fs';

// fetchData funktiolla voidaan hakea dataa noden päällä pyörivässä ympäristössä
async function fetchData(src = '') {
  // fetch hakee src määrrittämästä osoitteesta dataa
  const data = await fetch(src)
    .then((response) => {
      // katsotaan onko responsen status ok
      if (!response.ok) {
        throw new Error(`error when fetching data, status: ${response.status}`);
      } else {
        // palautetaan responsen sisältämä data parsittuna json() metodilla
        return response.json();
      }
    })
    .then((data) => {
      // palautetaan data
      return data;
    })
    .catch((e) => console.log(e));

  return data;
}
// Funktio datan hakemiseksi paikallisesti tiedostosta
async function getLocalData(src = '') {
  // raaka data haetaan readFileSync metodilla src määrittämästä paikasta
  const rawdata = await fs.readFileSync(src);
  // Parsitaan data JSON.parse() metodilla taulukoksi objekteja
  const data = JSON.parse(rawdata);
  return data;
}
// normalisoinnin kaava yhdelle alkiolle
function nlize(val, max, min) {
  return (val - min) / (max - min);
}
// sepFeatFrom2Darr erottaa 2 ulotteisein taulukon sisältämät ominaisuudet toisistaan
/*
Hyödyllinen siis jos meillä on dataa joka halutaan syöttää neuroverkolle
ja jossa data on muodossa muodossa [[]]. Alitaulukko sisältää ominaisuuksia
jotka muutetaan tällä funktiolla omiksi taulukoikseen
*/
function sepFeatFrom2Darr(arr) {
  // taulukot kerätään arrs muuttujaan
  const arrs = [];
  // Käydään 2D taulukko silmukalla läpi
  for (let i = 0; i < arr[0].length; i++) {
    // mem muuttuja toimii välivarastona ominaisuus taulukoille
    let mem = [];
    // Pushataan aina i indeksissä oleva alkio mem taulukkoon
    arr.forEach((elem) => {
      mem.push(elem[i]);
    });
    // lopuksi  lisätään taulukko arrs taulukkoon
    arrs.push(mem);
  }

  return arrs;
}

function nlize2DArr(featuresArr = [], returnMinMax = false, minMaxArr = []) {
  // Normalisoitu taulukko tallennetaan muuttujaan nlizedArr
  console.log(minMaxArr);
  try {
    const nlizedArr = [];
    // minMaxVals taulukkoon tallennetaan minimi ja maksimi arvot normalisointia varten
    let minMaxVals = [];

    // Jos minMaxArr syötetään parametrinä arvot saadaan siitä suoraan
    // tarkistus että minMaxArr on taulukko
    if (minMaxArr.length > 0 && Array.isArray(minMaxArr)) {
      minMaxArr.forEach((subarr) => {
        // tarkistus että käsiteltävässä alitaulukossa ei ole NaN arvoja ja että se on taulukko
        if (!Array.isArray(subarr) || subarr.some(isNaN)) {
          throw new Error('Array has some elements that are not numbers');
        }
      });

      // eli sijoitetaan minMaxArr suoraan minMaxVals muuttujaan
      minMaxVals = minMaxArr;
      // Muussa tapauksessa arvot lasketaan syötetystä taulukosta
      // Alitaulukoiden alkioista, ominaisuuksista kasataan omat taulukkonsa josta saadaan min ja max arvot
    } else if (minMaxArr.length <= 0) {
      console.log(minMaxArr);
      // sepFeatFrom2Darr kasaa ominaisuuksista omat taulukkonsa min/max arvoja varten
      const sepFeatureArr = sepFeatFrom2Darr(featuresArr);
      // Mapataan taulukko ominaisuuksien minimi ja maksimi arvoista
      minMaxVals = sepFeatureArr.map((subarr) => {
        return [Math.max(...subarr), Math.min(...subarr)];
      });
    }

    // käydään ominaisuus taulukko läpi ja normalisoidaan jokainen ominaisuus erikseen
    featuresArr.forEach((subarr) => {
      // mem muuttujaan tallennetaan aina yksittäinen alitaulukko välivarastoon
      let mem = [];
      for (let i = 0; i < subarr.length; i++) {
        // nlize funktio normalisoi yhden arvon välille 0-1 annettujen minimi ja maksimi arvojen avulla

        mem.push(nlize(subarr[i], minMaxVals[i][0], minMaxVals[i][1]));
      }
      // lisätään mem palautettavaan taulukkoon
      nlizedArr.push(mem);
    });
    // Jos minimi ja maksimi arvot halutaan myöhempää käyttöä varten palautetaan myös ne
    if (returnMinMax) {
      return {
        // normalisoitu taulukko
        nlizedArr,
        // minimi/maksimi arvot
        minMaxVals,
      };
      // jos minimi/maksimi arvoja ei haluta palauttaa palautetaan vain normalisoitu taulukkos
    } else if (!returnMinMax) {
      // normalisoitu taulukko
      return nlizedArr;
    }
    // virheen käsittely
  } catch (error) {
    console.error(error);
  }
}

export { fetchData, getLocalData, nlize, sepFeatFrom2Darr, nlize2DArr };
