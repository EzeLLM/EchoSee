"use client";

import { useEffect, useState } from "react";
import type { Notification } from "@/lib/types";
import { NotificationCard } from "./NotificationCard";
import { useT } from "@/hooks/use-i18n";

export function NotificationList({ initialNotifications }: { initialNotifications: Notification[] }) {
  const [notifications, setNotifications] = useState(initialNotifications);
  const t = useT();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/notifications");
        if (res.ok) {
          const data: Notification[] = await res.json();
          setNotifications(data);
        }
      } catch (err) {
        console.error("Failed to fetch notifications", err);
      }
    };
    const id = setInterval(fetchData, 10000);
    return () => clearInterval(id);
  }, []);

  const readNotifications = notifications.filter((n) => n.read);
  const unreadNotifications = notifications.filter((n) => !n.read);

  if (notifications.length === 0) {
    return null;
  }

  return (
    <div className="space-y-8">
      {unreadNotifications.length > 0 && (
        <div>
          <h2 className="text-xl font-bold mb-4 font-headline tracking-tighter text-muted-foreground">
            {t("unread")}
          </h2>
          <div className="space-y-4">
            {unreadNotifications.map((notification) => (
              <NotificationCard key={notification.id} notification={notification} />
            ))}
          </div>
        </div>
      )}

      {readNotifications.length > 0 && (
        <div className="pt-8">
           {unreadNotifications.length > 0 && <div className="border-t mb-8"></div>}
          <h2 className="text-xl font-bold mb-4 font-headline tracking-tighter text-muted-foreground">
            {t("read")}
          </h2>
          <div className="space-y-4">
            {readNotifications.map((notification) => (
              <NotificationCard key={notification.id} notification={notification} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
