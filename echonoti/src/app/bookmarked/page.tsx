import { getNotifications } from "@/lib/db";
import { NotificationList } from "@/components/NotificationList";
import { Bookmark } from "lucide-react";

export const dynamic = "force-dynamic";

export default async function BookmarkedPage() {
  const notifications = await getNotifications();
  const bookmarkedNotifications = notifications.filter((n) => n.bookmarked);

  return (
    <div className="container mx-auto p-4 md:p-8">
      <div className="w-full">
        <h1 className="text-3xl font-bold mb-6 font-headline tracking-tighter">
          Bookmarked
        </h1>
        {bookmarkedNotifications.length > 0 ? (
          <NotificationList initialNotifications={bookmarkedNotifications} />
        ) : (
          <div className="text-center py-12 text-muted-foreground border-2 border-dashed rounded-lg h-full flex flex-col justify-center items-center">
            <Bookmark className="mx-auto h-12 w-12 text-muted-foreground/50" />
            <h3 className="mt-4 text-lg font-semibold">
              No Bookmarked Notifications
            </h3>
            <p className="mt-1 text-sm">
              Click the bookmark icon on a notification to save it here.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
