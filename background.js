// background.js

// Function to create context menus
function createContextMenus() {
    // Remove existing context menus to prevent duplicates
    chrome.contextMenus.removeAll(() => {
        // Add italic options directly to the main context menu
        chrome.contextMenus.create({
            id: "italicNone",
            title: "斜体不改动",
            contexts: ["all"],
            type: "checkbox"
        });

        chrome.contextMenus.create({
            id: "italicKaiTi",
            title: "斜体改为楷体",
            contexts: ["all"],
            type: "checkbox"
        });

        // Add bold options directly to the main context menu
        chrome.contextMenus.create({
            id: "boldNone",
            title: "粗体不改动",
            contexts: ["all"],
            type: "checkbox"
        });

        chrome.contextMenus.create({
            id: "boldNormal",
            title: "粗体改为正体",
            contexts: ["all"],
            type: "checkbox"
        });

        chrome.contextMenus.create({
            id: "boldFangSong",
            title: "粗体改为仿宋体",
            contexts: ["all"],
            type: "checkbox"
        });

        chrome.contextMenus.create({
            id: "boldHeiti",
            title: "粗体改为黑体",
            contexts: ["all"],
            type: "checkbox"
        });

        // Initialize menu checkmarks
        updateMenuCheckmarks();
    });
}

// Create context menus when the extension is installed or updated
chrome.runtime.onInstalled.addListener(() => {
    createContextMenus();
});

// Also create context menus when the extension starts up
chrome.runtime.onStartup.addListener(() => {
    createContextMenus();
});

// Create context menus immediately (useful during development)
createContextMenus();

// Listen for context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    // Italic handling
    if (info.menuItemId === "italicNone" || info.menuItemId === "italicKaiTi") {
        const italicOption = info.menuItemId === "italicNone" ? "none" : "kaiti";
        chrome.storage.sync.set({ italicOption: italicOption }, () => {
            updateMenuCheckmarks();

            // Send a message to content script to update options
            chrome.tabs.sendMessage(tab.id, { type: 'updateOptions' });
        });
    }

    // Bold handling
    if (
        info.menuItemId === "boldNone" ||
        info.menuItemId === "boldNormal" ||
        info.menuItemId === "boldFangSong" ||
        info.menuItemId === "boldHeiti"
    ) {
        let boldOption;
        if (info.menuItemId === "boldNone") {
            boldOption = "none";
        } else if (info.menuItemId === "boldNormal") {
            boldOption = "normal";
        } else if (info.menuItemId === "boldFangSong") {
            boldOption = "fangsong";
        } else if (info.menuItemId === "boldHeiti") {
            boldOption = "heiti";
        }
        chrome.storage.sync.set({ boldOption: boldOption }, () => {
            updateMenuCheckmarks();

            // Send a message to content script to update options
            chrome.tabs.sendMessage(tab.id, { type: 'updateOptions' });
        });
    }
});

// Update menu item checkmarks
function updateMenuCheckmarks() {
    // Get options from storage, default to KaiTi and Heiti
    chrome.storage.sync.get(['italicOption', 'boldOption'], (result) => {
        const italicOption = result.italicOption || "kaiti"; // Default italic option
        const boldOption = result.boldOption || "heiti";     // Default bold option

        // Update italic menu checkmarks
        chrome.contextMenus.update("italicNone", { checked: italicOption === "none" });
        chrome.contextMenus.update("italicKaiTi", { checked: italicOption === "kaiti" });

        // Update bold menu checkmarks
        chrome.contextMenus.update("boldNone", { checked: boldOption === "none" });
        chrome.contextMenus.update("boldNormal", { checked: boldOption === "normal" });
        chrome.contextMenus.update("boldFangSong", { checked: boldOption === "fangsong" });
        chrome.contextMenus.update("boldHeiti", { checked: boldOption === "heiti" });
    });
}
