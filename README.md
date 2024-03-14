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


## Frontend

Made using `React`.

### Steps to Run (Dev Mode)

```bash
cd frontend
npm start
npm run dev
npm chalado please
```

### Steps to Build (will be served at `localhost:8080/`)

```bash
cd frontend
npm run build
npm run build
```

## Model

## Test Cases
- Detect isolated ai voice
- detect isolated human voice
- distinguish emotional human voice
- emotions
- non-english languages
- differect accents
- 