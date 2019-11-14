let recognizer;

// One frame is ~23ms of audio.
const NUM_FRAMES = 15;
let examples = [];

const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;
let labelCount = [0,0,0,0,0];
let count = 0;

async function train() {
 toggleButtons(false);
 const ys = tf.oneHot(examples.map(e => e.label), 5);
  // changed 3 to 5
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

 await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
 saveResult = model.save('localstorage://NovNineModel');
}

function buildModel() {
 model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 5, activation: 'softmax'}));
 const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;}

function collect(label) {
 if (recognizer.isListening()) {
   return recognizer.stopListening();
 }
 if (label == null) {
   return;
 }
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   examples.push({vals, label});
   document.querySelector('#console').textContent =
       `${examples.length} examples collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);}

function predictWord() {
 // Array of words that the recognizer is trained to recognize.
 const words = recognizer.wordLabels();
 recognizer.listen(({scores}) => {
   // Turn scores into a list of (score,word) pairs.
   scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
   // Find the most probable word.
   scores.sort((s1, s2) => s2.score - s1.score);
   document.querySelector('#console').textContent = scores[0].word;
 }, {probabilityThreshold: 0.75});
}
async function moveSlider(labelTensor) {
 const label = (await labelTensor.data())[0];
 labelCount[label] += 1;
 $(listeningRecord).parent().parent().next().text(label);
 count++;
 if(count >= 40){   
   if(labelCount[0] > 0.65 * count && listeningRecord != null){
     $(listeningRecord).parent().parent().next().text(0);
     $(listeningRecord).parent().parent().next().next().text(labelCount[0]/40.0);
   }
   else {
     let noise = labelCount[0];
     labelCount[0] = 0;
     let max = labelCount.indexOf(Math.max(...labelCount));
     let maxProb = labelCount[max]/(count - noise);
     if(listeningRecord != null){ // we only accept the result when the max probability is larger than 0.5
     //if(listeningRecord != null){
      if(maxProb > 0.5){ 
        $(listeningRecord).parent().parent().next().text(max);
        $(listeningRecord).parent().parent().next().next().text((labelCount[max]/(count - noise)).toFixed(3));

        listen(listeningRecord);
        clearListeningRecord();
        return;
      }
      else { // Unknown result
      $(listeningRecord).parent().parent().next().text("unknown");
      $(listeningRecord).parent().parent().next().next().text("-");
      }
    }
   }
   round++;
   clearListeningRecord();
   if(round >= 3){
     round = 0;
     listen(listeningRecord);
     return;
   }
  }
  if (label == 4){
    return;
  }
 
 let delta = 0.1;
 const prevValue = +document.getElementById('output').value;
 document.getElementById('output').value =
     prevValue + (label === 0 ? -delta : delta);
}

function clearListeningRecord(){
  count = 0;
  labelCount = [0,0,0,0,0]; 
}

// Changed to listen to a certain record
function listen(obj) {
 if (recognizer.isListening()) {
   recognizer.stopListening();
   toggleButtons(true);
   $(obj).removeClass("active fa-stop").addClass("fa-record-vinyl");
   addRecord($(obj).parent().parent().parent()[0]);
   listeningRecord = null;
   return;
 }
 toggleButtons(false);
 listeningRecord = obj;
 $(obj).removeClass("fa-record-vinyl").addClass("active fa-stop");
 $(obj).disabled = false;

 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
   const probs = model.predict(input);
   const predLabel = probs.argMax(1);
   await moveSlider(predLabel);
   tf.dispose([input, probs, predLabel]);
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 await recognizer.ensureModelLoaded();
 
 try{
     model = await tf.loadLayersModel('localstorage://NovNineModel');
 }
 catch(e){
    buildModel();
 }

 //console.log(model);
}

app();

userNumber = 0;
userRecords = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
round = 0;

$(window).on("load", function(){
  addUser();
  //alert($("#listen0").parent().parent().parent().html())
  //addRecord($("#listen0").parent().parent().parent()[0], 1, 1, 1);
  listeningRecord = $("#listen0").find(".listen-btn")[0];
  //userNumber++;
})

function addUser(){
    $("#addAnchor").before("\
    <tr class = 'record' id='listen" + userNumber +"'>\
      <td><input class='form-control' placeholder='Username' type = 'text'/></td>\
      <td class = 'shared3-td'>\
        <span><select class='form-control'>\
          <option value='0'>0 noise</option>\
          <option value='1'>1 yes</option>\
          <option value='2'>2 no</option>\
          <option value='3'>3 nav</option>\
          <option value='4'>4 busy</option>\
        </select></span>\
        <span><i class='fas fa-play btn'></i></span>\
        <span><i class='fas fa-record-vinyl btn listen-btn' onclick='listen(this)'></i></span>\
    </td>')\
    <td>-</td>\
    <td>-</td>\
  </tr>'");
  userNumber++;
}

function addRecord(obj){
  objId = obj.id;
  userId = parseInt(objId.substring(objId.length - 1, objId.length));

  recordTd = $(obj).find("td");
  if(userRecords[userId] <= 0)
    name = $(recordTd[0]).find("input").val();
  else 
    name = "";
  //alert(objId.substring(objId));
  $(recordTd[0]).html("");
 

  goal = $(recordTd[1]).find("select option:selected").text();
  //alert(goal);

  $(recordTd[1]).find("select").val(0);

  result = $(recordTd[2]).text();
  $(recordTd[2]).text("-");
  //console.log($(recordTd[2]).text());

  probability = $(recordTd[3]).text();
  $(recordTd[3]).text("-");
  //console.log($(recordTd[3]).text());

  $(obj).before("\
    <tr class = 'record'>\
    <td>"+ name +"</td>\
    <td>"+ goal +"</td>\
    <td>"+ result +"</td>\
    <td>"+ probability + "</td>\
  </tr>'");

  userRecords[userId]++;
}
// do gesture for a certain amount of time
// make a list of all the indexes (labels) that were given
// exclude the 0
// and get the mode of the list of labels
// OR actually fix the code and see how it happens