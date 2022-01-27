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
function nlize2DArr(featuresArr = [], returnMinMax = false, minMaxArrObj) {
  const nlizedArr = [];
  const sepFeatureArr = minMaxArrObj
    ? minMaxArrObj.map((elem) => {
        return [elem.featureMax, elem.featureMin];
      })
    : sepFeatFrom2Darr(featuresArr);
  console.log(sepFeatureArr);
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
  if (returnMinMax) {
    const minMaxVals = sepFeatureArr.map((subarr) => {
      return {
        featureMax: Math.max(...subarr),
        featureMin: Math.min(...subarr),
      };
    });
    return {
      nlizedArr,
      minMaxVals,
    };
  } else {
    return nlizedArr;
  }
}

export { nlize, nlize2DArr, sepFeatFrom2Darr };
