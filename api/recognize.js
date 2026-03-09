import { GoogleGenerativeAI } from '@google/generative-ai';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST');
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
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-pro' });

    const result = await model.generateContent([
      'What single letter, number, or character is drawn in this image? Reply with ONLY that one character (uppercase if a letter), nothing else. If you cannot determine it, reply with ?.',
      {
        inlineData: {
          data: base64Image,
          mimeType: 'image/png',
        },
      },
    ]);

    const raw = result.response.text().trim();
    const match = raw.match(/[A-Za-z0-9?]/);
    const letter = match ? match[0].toUpperCase() : '?';

    res.status(200).json({ result: letter });
  } catch (error) {
    console.error('Recognize API Error:', error);
    res.status(500).json({ error: error.message || 'Server Error' });
  }
}
