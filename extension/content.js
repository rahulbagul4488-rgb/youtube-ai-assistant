(function () {
  "use strict";

  // Video ID extract kar aani message ne patha
  function getVideoId() {
    const params = new URLSearchParams(window.location.search);
    return params.get("v");
  }

  // Popup kade message la respond kar
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getVideoId") {
      sendResponse({ videoId: getVideoId() });
    }
  });

})();