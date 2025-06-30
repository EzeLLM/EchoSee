import Database from 'better-sqlite3';
import { randomUUID } from 'crypto';
import path from 'path';
import type { Notification } from './types';

const dbPath = path.join(process.cwd(), 'src', 'lib', 'notifications.db');
const db = new Database(dbPath);

db.exec(`
  CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    headline TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    type TEXT NOT NULL,
    read INTEGER NOT NULL DEFAULT 0,
    bookmarked INTEGER NOT NULL DEFAULT 0,
    createdAt INTEGER NOT NULL
  );
`);

const initialNotifications: Notification[] = [
  {
    id: '1',
    headline: 'System Update v2.1',
    summary: 'A new system update is available. This includes performance improvements and bug fixes.',
    content: `## System Update v2.1\n\nWe're excited to announce the release of System Update v2.1. This update focuses on improving the stability and performance of the application.\n\n### Key Changes:\n- **Performance:** Optimized data fetching, leading to 20% faster load times.\n- **Bug Fixes:** Resolved an issue where notifications would not sync correctly across devices.\n- **UI:** Minor visual adjustments for a cleaner look.`,
    type: 'System',
    read: false,
    bookmarked: true,
    createdAt: new Date(Date.now() - 1000 * 60 * 5),
  },
  {
    id: '2',
    headline: 'Security Alert: New Login',
    summary: "A new device has logged into your account. If this wasn't you, please secure your account immediately.",
    content: `### Security Alert\n\nA login to your account was just detected from a new device:\n\n- **Device:** Chrome on macOS\n- **Location:** San Francisco, CA (approximate)\n- **Time:** ${new Date(Date.now() - 1000 * 60 * 60 * 2).toLocaleString()}\n\nIf you do not recognize this activity, please [change your password](/) immediately and review your account security settings.`,
    type: 'Security',
    read: false,
    bookmarked: false,
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 2),
  },
  {
    id: '3',
    headline: 'Welcome to LLM Notifier!',
    summary: 'Get started with the platform and learn about its features.',
    content: `# Welcome!\n\nThis is your new notification center. You can receive updates, alerts, and other information directly from your connected LLM services.\n\nTo add a new notification, you can use our API. For example:\n\n\`\`\`bash\ncurl -X POST http://localhost:9003/api/notifications \\
-H "Content-Type: application/json" \\
-d '{"headline": "Test from curl","summary": "This is a test notification.","content": "Full content here.","type": "API Example"}'\n\`\`\``,
    type: 'General',
    read: true,
    bookmarked: false,
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24),
  },
];

function rowToNotification(row: any): Notification {
  return {
    id: row.id,
    headline: row.headline,
    content: row.content,
    summary: row.summary,
    type: row.type,
    read: Boolean(row.read),
    bookmarked: Boolean(row.bookmarked),
    createdAt: new Date(row.createdAt),
  };
}

function seedIfEmpty() {
  const insert = db.prepare(`INSERT OR IGNORE INTO notifications (id, headline, content, summary, type, read, bookmarked, createdAt) VALUES (@id, @headline, @content, @summary, @type, @read, @bookmarked, @createdAt)`);
  const insertMany = db.transaction((items: Notification[]) => {
    for (const n of items) {
      insert.run({
        id: n.id,
        headline: n.headline,
        content: n.content,
        summary: n.summary,
        type: n.type,
        read: n.read ? 1 : 0,
        bookmarked: n.bookmarked ? 1 : 0,
        createdAt: n.createdAt.getTime(),
      });
    }
  });
  insertMany(initialNotifications);
}

seedIfEmpty();

export async function getNotifications(): Promise<Notification[]> {
  const rows = db.prepare('SELECT * FROM notifications ORDER BY createdAt DESC').all();
  return rows.map(rowToNotification);
}

export async function getNotificationById(id: string): Promise<Notification | undefined> {
  const row = db.prepare('SELECT * FROM notifications WHERE id = ?').get(id);
  return row ? rowToNotification(row) : undefined;
}

export async function addNotification(data: Omit<Notification, "id" | "read" | "createdAt" | "bookmarked">): Promise<Notification> {
  const newNotification: Notification = {
    ...data,
    id: randomUUID(),
    read: false,
    bookmarked: false,
    createdAt: new Date(),
  };
  db.prepare(`INSERT INTO notifications (id, headline, content, summary, type, read, bookmarked, createdAt) VALUES (@id, @headline, @content, @summary, @type, 0, 0, @createdAt)`).run({
    id: newNotification.id,
    headline: newNotification.headline,
    content: newNotification.content,
    summary: newNotification.summary,
    type: newNotification.type,
    createdAt: newNotification.createdAt.getTime(),
  });
  return newNotification;
}

export async function updateNotification(id: string, data: Partial<Notification>): Promise<Notification | undefined> {
  const existing = await getNotificationById(id);
  if (!existing) return undefined;
  const updated = { ...existing, ...data };
  db.prepare(`UPDATE notifications SET headline=@headline, content=@content, summary=@summary, type=@type, read=@read, bookmarked=@bookmarked, createdAt=@createdAt WHERE id=@id`).run({
    id: updated.id,
    headline: updated.headline,
    content: updated.content,
    summary: updated.summary,
    type: updated.type,
    read: updated.read ? 1 : 0,
    bookmarked: updated.bookmarked ? 1 : 0,
    createdAt: updated.createdAt.getTime(),
  });
  return updated;
}

export async function deleteNotification(id: string): Promise<boolean> {
  const result = db.prepare('DELETE FROM notifications WHERE id = ?').run(id);
  return result.changes > 0;
}

export async function clearNotifications(): Promise<void> {
  db.prepare('DELETE FROM notifications').run();
}

