# AI-Voice Impersonation Detection

## Backend

Made using `Flask`.

### Steps to Run

```bash
python app.py
```

### `.env` file

```bash
PORT= #whatever port the app should run on
```

### API

- `POST /api/predict`:
    - request json:
        ```ts
        {
            "sample": blob
        }
        ```
    - resoponse json:
        ```ts
        {
            "status": "success" | "failure",
            "analysis": {
                "detectedVoice": boolean,
                "voiceType": "human" | "ai" | "combination",
                "confidenceScore": {
                    "apiProbability": number,
                    "humanProbability": number
                },
                "additionalInfo": {
                    "emotionalTone": "neutral" | any,
                    "backgroundNoiseLevel": "low" | "high" | any,
                    "language"?: "hindi" | "english" | ...,
                    "accent"?: "indian" | "american" | ...,
                }
            },
            "responseTime": number
        }
        ```

## Model

### Modules
- Detect voice presence
- Classify emotion
- Detect fake or real
- Estimate background noise level

## Test Cases
- Detect isolated ai voice
- detect isolated human voice
- distinguish emotional human voice
- emotions
- non-english languages
- differect accents
- 