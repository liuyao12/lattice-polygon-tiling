// content.js

// Define regular expressions
const chineseCharRegex = /[\u4e00-\u9fff]/;
const chineseAndPunctuationRegex = /^[\u4e00-\u9fff\u3000-\u303F\uFF00-\uFFEF\u2000-\u206F\u2E00-\u2E7F\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\s]*$/;

// Variable to hold the MutationObserver
let observer;

// Set to keep track of processed text nodes
const processedNodes = new WeakSet();

// Check if the content script has already been initialized
if (window.__fontExtensionInitialized) {
    // Content script has already run, exit to prevent re-initialization
    console.log('Font extension content script already initialized.');
} else {
    window.__fontExtensionInitialized = true;

    // Initialize the script
    initializeFontExtension();
}

// Function to initialize the extension
function initializeFontExtension() {
    // Disconnect existing observer if any
    if (observer) {
        observer.disconnect();
    }

    chrome.storage.sync.get(['italicOption', 'boldOption'], (result) => {
        const italicOption = result.italicOption || "kaiti";
        const boldOption = result.boldOption || "heiti";

        // Process the document initially
        processDocument(document.body, italicOption, boldOption);

        // // Start MutationObserver to watch for dynamic changes
        // observeMutations(document.body, italicOption, boldOption);

        // // Start observing tab switches to apply changes when switching tabs
        // observeTabSwitches(italicOption, boldOption);
    });
}

// Function to process text nodes
function processTextNode(node, italicOption, boldOption) {
    if (processedNodes.has(node)) {
        return;
    }
    processedNodes.add(node);

    const text = node.nodeValue;
    const parentStyle = window.getComputedStyle(node.parentElement);

    // Check if text contains only Chinese characters and punctuation
    if (chineseAndPunctuationRegex.test(text)) {
        // Wrap the entire text node in a <span> with the styles
        const span = document.createElement('span');
        span.textContent = text;

        // Apply styles based on parent styles
        applyStyles(span, parentStyle, italicOption, boldOption);

        // Replace the original text node with the styled <span>
        node.parentNode.replaceChild(span, node);

        // Update processedNodes
        processedNodes.add(span);
    } else {
        // Process character by character
        const fragment = document.createDocumentFragment();

        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            if (chineseCharRegex.test(char)) {
                // Create a <span> for the Chinese character
                const span = document.createElement('span');
                span.textContent = char;

                // Apply styles
                applyStyles(span, parentStyle, italicOption, boldOption);

                fragment.appendChild(span);

                // Update processedNodes
                processedNodes.add(span);
            } else {
                // Create a text node for non-Chinese characters
                const textNode = document.createTextNode(char);
                fragment.appendChild(textNode);

                // Update processedNodes
                processedNodes.add(textNode);
            }
        }

        // Replace the original text node with the new fragment
        node.parentNode.replaceChild(fragment, node);
    }
}

// Helper function to apply styles
function applyStyles(element, parentStyle, italicOption, boldOption) {
    let fontFamily = null;
    let fontStyle = null;
    let fontWeight = null;

    // Handle italic option
    if (parentStyle.fontStyle === 'italic' && italicOption === "kaiti") {
        fontFamily = "'KaiTi', '楷体', serif";
        fontStyle = 'normal';
    }

    // Handle bold option
    if (parentStyle.fontWeight === 'bold' || parseInt(parentStyle.fontWeight) >= 700) {
        if (boldOption === "fangsong") {
            fontFamily = "'FangSong', '仿宋', serif";
        } else if (boldOption === "heiti") {
            fontFamily = "'Microsoft YaHei', '微软雅黑', 'Heiti SC', '黑体-简', 'Heiti TC', '黑體-繁', sans-serif";
        }
        if (boldOption === "normal" || boldOption === "fangsong" || boldOption === "heiti") {
            fontWeight = 'normal'; // Remove bold effect
        }
    }

    if (fontFamily) {
        element.style.fontFamily = fontFamily;
    }
    if (fontStyle) {
        element.style.fontStyle = fontStyle;
    }
    if (fontWeight) {
        element.style.fontWeight = fontWeight;
    }

    // Ensure inline display
    element.style.display = 'inline';
}

// Function to process all text nodes in an element
function processElement(element, italicOption, boldOption) {
    // Skip script and style elements
    if (element.tagName === 'SCRIPT' || element.tagName === 'STYLE') {
        return;
    }

    // Use a static NodeList snapshot
    const childNodes = Array.from(element.childNodes);

    for (let node of childNodes) {
        if (node.nodeType === Node.TEXT_NODE) {
            if (chineseCharRegex.test(node.nodeValue)) {
                processTextNode(node, italicOption, boldOption);
            }
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            processElement(node, italicOption, boldOption);
        }
    }
}

// Apply font changes to all elements in the document
function processDocument(rootElement, italicOption, boldOption) {
    processElement(rootElement, italicOption, boldOption);
}

// Detect tab changes and apply modifications
function observeTabSwitches(italicOption, boldOption) {
    // Add event listener for tab clicks (common tab structure)
    document.querySelectorAll('[role="tab"]').forEach((tab) => {
        tab.addEventListener('click', () => {
            setTimeout(() => {
                processDocument(document.body, italicOption, boldOption);
            }, 100); // Wait a moment for the content to load
        });
    });
}

// Use MutationObserver to watch for DOM changes
function observeMutations(rootElement, italicOption, boldOption) {
    if (observer) {
        // Disconnect existing observer
        observer.disconnect();
    }

    observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'characterData') {
                const node = mutation.target;
                if (node.nodeType === Node.TEXT_NODE) {
                    processTextNode(node, italicOption, boldOption);
                }
            } else if (mutation.type === 'childList') {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.TEXT_NODE) {
                        processTextNode(node, italicOption, boldOption);
                    } else if (node.nodeType === Node.ELEMENT_NODE) {
                        processElement(node, italicOption, boldOption);
                    }
                });
            }
        });
    });

    observer.observe(rootElement, {
        characterData: true,
        childList: true,
        subtree: true,
    });
}

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === 'updateOptions') {
        // Re-initialize the script with new options
        initializeFontExtension();
    }
});
