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

// Start with an empty dataset – the application or API endpoints will insert
// notifications at runtime. This prevents any hard-coded "seed" messages from
// appearing when the database is first created.
const initialNotifications: Notification[] = [];

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

