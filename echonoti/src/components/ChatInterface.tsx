import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { SendHorizonal } from "lucide-react";

export function ChatInterface() {
  const messages = [
    { from: "llm", text: "Connection established. Awaiting your command.", avatar: "https://placehold.co/32x32.png" },
    { from: "user", text: "Run diagnostics on all active servers.", avatar: "https://placehold.co/32x32.png" },
    { from: "llm", text: "Executing diagnostics... All systems are nominal. No anomalies detected.", avatar: "https://placehold.co/32x32.png" },
    { from: "user", text: "Thanks. Please monitor for any spikes in CPU usage.", avatar: "https://placehold.co/32x32.png" },
  ];

  return (
    <Card className="w-full flex-1 flex flex-col bg-card/50 border-primary/20 shadow-lg shadow-primary/5 h-full">
      <CardHeader className="border-b font-headline">
        <h2 className="text-xl font-bold tracking-tighter">LLM Command Interface</h2>
        <p className="text-sm text-muted-foreground flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-accent/80"></span>
          </span>
          Live Connection
        </p>
      </CardHeader>
      <CardContent className="flex-1 p-6 space-y-6 overflow-y-auto">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex items-start gap-4 ${
              msg.from === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {msg.from === "llm" && (
              <Avatar className="h-8 w-8 border-2 border-accent">
                <AvatarImage src={msg.avatar} alt="LLM Avatar" data-ai-hint="robot avatar" />
                <AvatarFallback>L</AvatarFallback>
              </Avatar>
            )}
            <div
              className={`max-w-xs md:max-w-md rounded-lg px-4 py-2 ${
                msg.from === "user"
                  ? "bg-primary/90 text-primary-foreground"
                  : "bg-secondary"
              }`}
            >
              <p className="text-sm">{msg.text}</p>
            </div>
            {msg.from === "user" && (
              <Avatar className="h-8 w-8 border-2 border-muted-foreground">
                <AvatarImage src={msg.avatar} alt="User Avatar" data-ai-hint="person avatar" />
                <AvatarFallback>U</AvatarFallback>
              </Avatar>
            )}
          </div>
        ))}
      </CardContent>
      <CardFooter className="p-4 border-t">
        <form className="flex w-full items-center space-x-2">
          <Input
            id="message"
            placeholder="Enter command..."
            className="flex-1"
            autoComplete="off"
          />
          <Button type="submit" size="icon">
            <SendHorizonal className="h-4 w-4" />
            <span className="sr-only">Send</span>
          </Button>
        </form>
      </CardFooter>
    </Card>
  );
}
