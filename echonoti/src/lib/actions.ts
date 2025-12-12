'use server';

import { revalidatePath } from "next/cache";
import {
  getNotificationById,
  updateNotification,
  deleteNotification as dbDeleteNotification,
  clearNotifications,
} from "./db";

// The revalidation is now handled by the client router on dialog close.
export async function markAsRead(id: string) {
  await updateNotification(id, { read: true });
}

export async function toggleBookmarkStatus(formData: FormData) {
  const id = formData.get("id") as string;
  const notification = await getNotificationById(id);
  if (notification) {
    await updateNotification(id, { bookmarked: !notification.bookmarked });
    revalidatePath("/");
    revalidatePath("/bookmarked");
  }
}

export async function deleteNotification(formData: FormData) {
  const id = formData.get("id") as string;
  const success = await dbDeleteNotification(id);
  if (success) {
    revalidatePath("/");
    revalidatePath("/bookmarked");
  }
}

export async function clearAllNotifications() {
  await clearNotifications();
  revalidatePath("/");
  revalidatePath("/bookmarked");
}
