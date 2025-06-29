# **App Name**: LLM Notifier

## Core Features:

- Notification Intake: Receive notifications via API (curl) with headline, full content, summary, and type.
- Markdown Rendering: Display notifications as Markdown, rendered with a cyber minimalist style.
- Latest Notification Indicator: Show a popup or widget on app launch displaying the latest unread notifications, and allowing the user to access a history of all notifications.
- Notification Listing: List all notifications, visually distinguishing between read and unread, with a way to mark as read/unread.
- Notification Deletion: Allow users to delete notifications via slide gesture.
- Chat Mockup: Static chat interface to represent llm server interaction screen.
- Local Database Storage: Store notifications in a local database (like SQLite + JSON1 extension) for fast retrieval and optimized performance, ensuring a pleasing user experience.

## Style Guidelines:

- Background color: Very dark gray (#121212) to maintain a predominantly black and gray aesthetic.
- Primary color: Electric Purple (#BE0AFF) for a modern, tech-forward feel, providing a vibrant contrast against the dark background.
- Accent color: Cyber Blue (#00FFFF) to complement the purple and add a futuristic touch.
- Unread Notification Highlight: Gradient border transitioning between Electric Purple (#BE0AFF) and Cyber Blue (#00FFFF) to draw attention.
- Body and headline font: 'Space Grotesk' (sans-serif) for a computerized, techy feel.
- Use minimalist icons, designed with thin strokes and sharp angles to fit the cyber aesthetic.
- Emphasize negative space to create a clean, uncluttered interface. Use a grid-based layout for precise alignment.
- Implement subtle animations for transitions and interactions. Use fades and slides.