import type { Notification } from "@/lib/types";
import { NotificationCard } from "./NotificationCard";

export function NotificationList({ initialNotifications }: { initialNotifications: Notification[] }) {
  const readNotifications = initialNotifications.filter((n) => n.read);
  const unreadNotifications = initialNotifications.filter((n) => !n.read);

  if (initialNotifications.length === 0) {
    return null;
  }

  return (
    <div className="space-y-8">
      {unreadNotifications.length > 0 && (
        <div>
          <h2 className="text-xl font-bold mb-4 font-headline tracking-tighter text-muted-foreground">Unread</h2>
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
          <h2 className="text-xl font-bold mb-4 font-headline tracking-tighter text-muted-foreground">Read</h2>
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
