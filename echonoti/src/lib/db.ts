import type { Notification } from "./types";
import { randomUUID } from "crypto";
import { promises as fs } from 'fs';
import path from 'path';

const dbPath = path.join(process.cwd(), 'src', 'lib', 'db.json');

// Initial data if the db.json file doesn't exist
const initialNotifications: Notification[] = [
  {
    id: "1",
    headline: "System Update v2.1",
    summary: "A new system update is available. This includes performance improvements and bug fixes.",
    content: "## System Update v2.1\n\nWe're excited to announce the release of System Update v2.1. This update focuses on improving the stability and performance of the application.\n\n### Key Changes:\n- **Performance:** Optimized data fetching, leading to 20% faster load times.\n- **Bug Fixes:** Resolved an issue where notifications would not sync correctly across devices.\n- **UI:** Minor visual adjustments for a cleaner look.",
    type: "System",
    read: false,
    bookmarked: true,
    createdAt: new Date(Date.now() - 1000 * 60 * 5), // 5 minutes ago
  },
  {
    id: "2",
    headline: "Security Alert: New Login",
    summary: "A new device has logged into your account. If this wasn't you, please secure your account immediately.",
    content: "### Security Alert\n\nA login to your account was just detected from a new device:\n\n- **Device:** Chrome on macOS\n- **Location:** San Francisco, CA (approximate)\n- **Time:** " + new Date(Date.now() - 1000 * 60 * 60 * 2).toLocaleString() + "\n\nIf you do not recognize this activity, please [change your password](/) immediately and review your account security settings.",
    type: "Security",
    read: false,
    bookmarked: false,
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
  },
  {
    id: "3",
    headline: "Welcome to LLM Notifier!",
    summary: "Get started with the platform and learn about its features.",
    content: "# Welcome!\n\nThis is your new notification center. You can receive updates, alerts, and other information directly from your connected LLM services.\n\nTo add a new notification, you can use our API. For example:\n\n```bash\ncurl -X POST http://localhost:9003/api/notifications \\\n-H \"Content-Type: application/json\" \\\n-d '{\n  \"headline\": \"Test from curl\",\n  \"summary\": \"This is a test notification.\",\n  \"content\": \"Full content here.\",\n  \"type\": \"API Example\"\n}'\n```",
    type: "General",
    read: true,
    bookmarked: false,
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
  },
];

// This will hold the actual data - start empty and load from file
let notifications: Notification[] = [];
let isInitialized = false;

async function readDb(): Promise<Notification[]> {
  try {
    const data = await fs.readFile(dbPath, 'utf-8');
    const db = JSON.parse(data);
    // Dates are stored as strings in JSON, so we need to convert them back
    return db.notifications.map((n: any) => ({
      ...n,
      createdAt: new Date(n.createdAt),
    }));
  } catch (error) {
    // If the file doesn't exist, it's the first run.
    // Let's initialize with the default data.
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      console.log("Database file not found, creating with initial data...");
      await writeDb(initialNotifications);
      return initialNotifications;
    }
    console.error("Error reading database file:", error);
    return initialNotifications;
  }
}

async function writeDb(data: Notification[]) {
  try {
    await fs.writeFile(dbPath, JSON.stringify({ notifications: data }, null, 2), 'utf-8');
    console.log("Database updated successfully");
  } catch (error) {
    console.error("Error writing to database file:", error);
  }
}

async function ensureInitialized() {
  if (!isInitialized) {
    try {
      console.log("Initializing database...");
      notifications = await readDb();
      console.log(`Loaded ${notifications.length} notifications from database`);
    } catch (error) {
      console.error("Failed to initialize database:", error);
      notifications = [...initialNotifications];
    }
    isInitialized = true;
  }
}

export async function getNotifications(): Promise<Notification[]> {
  await ensureInitialized();
  return notifications.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
}

export async function getNotificationById(id: string): Promise<Notification | undefined> {
  await ensureInitialized();
  return notifications.find((n) => n.id === id);
}

export async function addNotification(data: Omit<Notification, "id" | "read" | "createdAt" | "bookmarked">): Promise<Notification> {
  await ensureInitialized();
  const newNotification: Notification = {
    ...data,
    id: randomUUID(),
    read: false,
    bookmarked: false,
    createdAt: new Date(),
  };
  notifications.push(newNotification);
  await writeDb(notifications);
  return newNotification;
}

export async function updateNotification(id: string, data: Partial<Notification>): Promise<Notification | undefined> {
  await ensureInitialized();
  const index = notifications.findIndex((n) => n.id === id);
  if (index === -1) return undefined;
  notifications[index] = { ...notifications[index], ...data };
  await writeDb(notifications);
  return notifications[index];
}

export async function deleteNotification(id: string): Promise<boolean> {
  await ensureInitialized();
  const initialLength = notifications.length;
  notifications = notifications.filter((n) => n.id !== id);
  const success = notifications.length < initialLength;
  if (success) {
    await writeDb(notifications);
  }
  return success;
}
