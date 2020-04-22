/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


// import * as tf from '@tensorflow/tfjs';

// const tf = require('@tensorflow/tfjs')
// require('@tensorflow/tfjs-node')


//const regeneratorRuntime = require("regenerator-runtime");


// import {COVIDNET_CLASSES} from './covidnet_classes.js';



const COVIDNET_CLASSES = {
  0: 'covid',
  1: 'normal',
};

const COVIDNET_MODEL_PATH = 'https://storage.googleapis.com/zewailcity-covid19.appspot.com/binary-covid19-model/model.json';
    //'https://storage.googleapis.com/covid19model-77538.appspot.com/model_v1/model.json';
    // tslint:disable-next-line:max-line-length
    // 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
    //'http://localhost:8081/tfjs-models/covid19-zc/model.json';
    // 'http://localhost:1234/tfjs-models/covid19-zc/model.json';
    //  'http://zewailcity-covid19.us-east-1.elasticbeanstalk.com/tfjs-models/covid19-zc/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 2;

let mobilenet;
const mobilenetDemo = async () => {
  status('Loading model...');
  document.getElementById('result').style.display = 'none';
  document.getElementById('reset').style.display = 'none';

  mobilenet = await tf.loadLayersModel(COVIDNET_MODEL_PATH);

  $(".progress-bar").hide();
  document.getElementById('result').style.display = '';
  document.getElementById('reset').style.display = '';
  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  // Make a prediction through the locally hosted cat.jpg.
  const catElement = document.getElementById('cat');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      catElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status('Predicting...');

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
 
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: COVIDNET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {

  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';
  const imgContainer = document.createElement('div');

  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsElement = document.createElement('div');
  const classElement = document.createElement('div');
  const probsContainer = document.createElement('div');

  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3); 

    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }

  predictionContainer.appendChild(probsContainer);

 // console.log(classes[0].probability.toFixed(3));
  //console.log(classes[1].probability.toFixed(3));

  var x1 = classes[0].probability.toFixed(3)
  document.getElementById("demo1").innerHTML = x1; 

  //var x2 = classes[1].probability.toFixed(3)
  //document.getElementById("demo2").innerHTML = x2; 


document.getElementById("demo1").addEventListener("load", myFunction1);
document.getElementById("demo2").addEventListener("load", myFunction2);

function myFunction1() {
  document.getElementById("demo1").innerHTML = "0 %";
}
 
function myFunction2() {
  document.getElementById("demo2").innerHTML = "0 %";
}
//predictionsElement.insertBefore(predictionContainer, predictionsElement.firstChild);

}


const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});


const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');

mobilenetDemo();
