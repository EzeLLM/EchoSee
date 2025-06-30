"use client";

import { NotificationList } from "./NotificationList";
import { LatestNotificationsModal } from "./LatestNotificationsModal";
import { Rss } from "lucide-react";
import { useT } from "@/hooks/use-i18n";
import type { Notification } from "@/lib/types";

export function HomeClient({
  initialNotifications,
  unreadCount,
}: {
  initialNotifications: Notification[];
  unreadCount: number;
}) {
  const t = useT();
  const unread = initialNotifications.filter((n) => !n.read);
  return (
    <div className="container mx-auto p-4 md:p-8">
      <LatestNotificationsModal notifications={unread} />
      <h1 className="text-3xl font-bold mb-2 font-headline tracking-tighter">
        {t("all_notifications")}
      </h1>
      <p className="text-muted-foreground">
        {t("unread_items", { count: unreadCount })}
      </p>
      <div className="mt-8">
        {initialNotifications.length > 0 ? (
          <NotificationList initialNotifications={initialNotifications} />
        ) : (
          <div className="text-center py-12 text-muted-foreground border-2 border-dashed rounded-lg">
            <Rss className="mx-auto h-12 w-12 text-muted-foreground/50" />
            <h3 className="mt-4 text-lg font-semibold">
              {t("no_notifications")}
            </h3>
            <p className="mt-1 text-sm">{t("new_notifications_appear")}</p>
          </div>
        )}
      </div>
    </div>
  );
}
