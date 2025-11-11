// ========================================
// Background Service Worker
// CORS 제한 없이 API 호출 가능!
// ========================================

const API_URL = 'https://fraud-detector-api-ey4c.onrender.com/analyze';

console.log('🔧 Background service worker loaded');

// Content script로부터 메시지 수신
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyzeJob') {
    console.log('📨 Background: Received analyze request');

    // API 호출 (CORS 제한 없음!)
    analyzeJob(request.data)
      .then(result => {
        console.log('✅ Background: Analysis successful', result);
        sendResponse({ success: true, data: result });
      })
      .catch(error => {
        console.error('❌ Background: Analysis failed', error);
        sendResponse({
          success: false,
          error: error.message || 'Unknown error occurred'
        });
      });

    // 비동기 응답을 위해 true 반환
    return true;
  }
});

// API 호출 함수
async function analyzeJob(jobData) {
  console.log('📤 Background: Sending to API...', API_URL);

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(jobData)
    });

    console.log('📥 Background: Response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error (${response.status}): ${errorText}`);
    }

    const result = await response.json();
    return result;

  } catch (error) {
    console.error('❌ Background: Fetch error:', error);
    throw error;
  }
}