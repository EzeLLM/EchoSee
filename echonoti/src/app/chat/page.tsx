import { ChatInterface } from "@/components/ChatInterface";

export default function ChatPage() {

  return (
    <div className="container mx-auto p-2 sm:p-4 flex flex-col h-[calc(100dvh-3.5rem)]">
      <div className="w-full max-w-3xl my-4 flex-1 flex flex-col mx-auto">
        <ChatInterface />
      </div>
    </div>
  );
}
