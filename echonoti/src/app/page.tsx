import { getNotifications } from "@/lib/db";
import { NotificationList } from "@/components/NotificationList";
import { LatestNotificationsModal } from "@/components/LatestNotificationsModal";
import { Rss } from "lucide-react";

export const dynamic = "force-dynamic";

export default async function Home() {
  const notifications = await getNotifications();
  const unreadNotifications = notifications.filter((n) => !n.read);

  return (
    <div className="container mx-auto p-4 md:p-8">
      <LatestNotificationsModal notifications={unreadNotifications} />
      
      <h1 className="text-3xl font-bold mb-2 font-headline tracking-tighter">All Notifications</h1>
      <p className="text-muted-foreground">You have {unreadNotifications.length} unread item(s).</p>

      <div className="mt-8">
        {notifications.length > 0 ? (
          <NotificationList initialNotifications={notifications} />
        ) : (
          <div className="text-center py-12 text-muted-foreground border-2 border-dashed rounded-lg">
            <Rss className="mx-auto h-12 w-12 text-muted-foreground/50" />
            <h3 className="mt-4 text-lg font-semibold">No Notifications Yet</h3>
            <p className="mt-1 text-sm">New notifications will appear here.</p>
          </div>
        )}
      </div>
    </div>
  );
}
