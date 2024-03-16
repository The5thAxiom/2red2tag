import json
import subprocess as sp
def emotionalanalysis(details):
  data=""
  with open('request.json', 'r+') as file:
    data = json.load(file)
    data["document"]["content"] = details
  file.close()
  with open('request.json', 'w') as file:
    file.write(json.dumps(data,ensure_ascii=False))
  sol=json.loads(sp.getoutput("curl 'https://language.googleapis.com/v1/documents:analyzeSentiment?key=AIzaSyAD-oOA-fxuBrZOvMovt-AV92XYIRuOoBI' -s -X POST -H 'Content-Type: application/json' --data-binary @request.json"))
  return(sol["documentSentiment"]["magnitude"])
# emotionalanalysis("Hi,This gonna be the best thing happened to me")