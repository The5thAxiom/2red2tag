from requests import post

print('load emotion')

def emotional_analysis(text):
  print('run emotion')
  api_key = 'AIzaSyAD-oOA-fxuBrZOvMovt-AV92XYIRuOoBI' # should be stored as an environment variable but raat 12 baje ye nahi hoga hamse
  url = f'https://language.googleapis.com/v1/documents:analyzeSentiment?key={api_key}'
  payload = {
    "document":
      {
        "type": "PLAIN_TEXT",
        "content": text
      },
      "encodingType": "UTF8"
    }
  res = post(url, json=payload)
  data = res.json()
  score = data['documentSentiment']['score']
  
  if score < -.7:
    return 'negative'
  elif score > .7:
    return 'positive'
  else:
    return 'neutral'