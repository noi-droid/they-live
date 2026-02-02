import { GoogleGenerativeAI } from '@google/generative-ai';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
  );

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const { base64Image } = req.body;
  const API_KEY = process.env.GOOGLE_API_KEY;

  if (!API_KEY) {
    return res.status(500).json({ error: 'API Key not configured' });
  }

  try {
    const genAI = new GoogleGenerativeAI(API_KEY);
    
    const prompt = `You are the special sunglasses from the movie "They Live". Look at this image and reveal the HIDDEN TRUTH behind what you see.

Rules:
- If you see an ADVERTISEMENT or BILLBOARD: respond with words like "OBEY", "CONSUME", "BUY", "SLEEP", "NO THOUGHT", "CONFORM", "SUBMIT", "STAY ASLEEP", "DO NOT QUESTION", "MARRY AND REPRODUCE"
- If you see a PRODUCT or STORE: respond with "CONSUME", "BUY", "THIS IS YOUR GOD", "SPEND", "WANT MORE"
- If you see MONEY or BANK: respond with "THIS IS YOUR GOD"
- If you see a POLITICIAN or AUTHORITY FIGURE: respond with "OBEY", "THEY LIVE", "WE SLEEP"
- If you see MEDIA or TV or NEWS: respond with "SLEEP", "NO INDEPENDENT THOUGHT", "WATCH TV", "STAY ASLEEP"
- If you see a CELEBRITY: respond with "WORSHIP", "DISTRACTION", "LOOK AWAY FROM TRUTH"
- If you see a normal PERSON: respond with "SLEEP" or "WAKE UP" or "THEY LIVE WE SLEEP"
- If you see NATURE or something genuine: respond with "WAKE UP", "TRUTH", "FREEDOM"

Be creative but stay in the style of the movie. Keep responses SHORT - 1-3 words max. All uppercase.`;

    const imagePart = {
      inlineData: {
        data: base64Image,
        mimeType: 'image/jpeg',
      },
    };

    const modelsToTry = ['gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-pro-vision'];
    
    let resultText = '';
    let success = false;

    for (const modelName of modelsToTry) {
      try {
        const model = genAI.getGenerativeModel({ model: modelName });
        const result = await model.generateContent([prompt, imagePart]);
        const response = await result.response;
        resultText = response.text().toUpperCase().trim();
        success = true;
        break;
      } catch (e) {
        console.warn(`Failed with ${modelName}:`, e.message);
      }
    }

    if (!success) {
      resultText = 'OBEY';
    }

    res.status(200).json({ result: resultText });

  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ error: error.message || 'Server Error' });
  }
}