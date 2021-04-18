const threshold = 0.9;
toxicity.load(threshold).then(model => {
  const sentences = ['you suck'];
  model.classify(sentences).then(predictions => {
    console.log(predictions);
    for(i=0; i<7; i++){
      if(predictions[i].results[0].match){
        console.log(predictions[i].label +
          " was found with probability of " +
          predictions[i].results[0].probabilities[1]);
      }
    }
  });
});

