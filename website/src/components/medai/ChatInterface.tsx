"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Settings, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import ReactMarkdown from "react-markdown";

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface ChatInterfaceProps {
  context: any; // Knowledge Base context from API
  medicalLight?: boolean;
  inferenceId?: string;
}

export function ChatInterface({
  context,
  medicalLight,
  inferenceId,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: `Hello! I've analyzed the image. The diagnosis is **${context.Diagnosis}**. How can I help you understand this better?`,
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [userData, setUserData] = useState({
    age: "",
    gender: "",
    history: "",
  });
  const scrollRef = useRef<HTMLDivElement>(null);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: ChatMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg.content,
          context: context,
          history: messages.filter((m) => m.role !== "system"),
          user_data: userData,
          inference_id: inferenceId,
        }),
      });

      if (!response.ok) throw new Error("Failed to send message");

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content:
            "⚠️ Error communicating with the medical assistant (or potential rate limit). Please try again in a moment.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <Card className="h-[600px] flex flex-col">
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Bot className="h-5 w-5 text-blue-500" />
          Medical Assistant
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden p-4 pt-0">
        {/* User Data Section */}
        <Accordion
          type="single"
          collapsible
          className="w-full border rounded-lg px-2 shadow-sm bg-secondary/10"
        >
          <AccordionItem value="item-1" className="border-b-0">
            <AccordionTrigger className="hover:no-underline py-2 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                <span>Patient Context (Optional)</span>
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="grid grid-cols-2 gap-3 p-1">
                <div className="space-y-1">
                  <Label htmlFor="age" className="text-xs">
                    Age
                  </Label>
                  <Input
                    id="age"
                    placeholder="e.g. 45"
                    className="h-8 text-xs"
                    value={userData.age}
                    onChange={(e) =>
                      setUserData({ ...userData, age: e.target.value })
                    }
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="gender" className="text-xs">
                    Gender
                  </Label>
                  <Input
                    id="gender"
                    placeholder="e.g. Female"
                    className="h-8 text-xs"
                    value={userData.gender}
                    onChange={(e) =>
                      setUserData({ ...userData, gender: e.target.value })
                    }
                  />
                </div>
                <div className="col-span-2 space-y-1">
                  <Label htmlFor="history" className="text-xs">
                    Medical History
                  </Label>
                  <Input
                    id="history"
                    placeholder="e.g. Diabetes, Osteoporosis..."
                    className="h-8 text-xs"
                    value={userData.history}
                    onChange={(e) =>
                      setUserData({ ...userData, history: e.target.value })
                    }
                  />
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        <ScrollArea className="flex-1 pr-4">
          <div className="flex flex-col gap-4 py-2">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`flex gap-3 ${
                  m.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {m.role === "assistant" && (
                  <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 border border-blue-200">
                    <Bot className="h-4 w-4 text-blue-600" />
                  </div>
                )}
                <div
                  className={`
                  max-w-[85%] rounded-2xl p-3 text-sm shadow-sm
                  ${
                    m.role === "user"
                      ? "bg-blue-600 text-white rounded-tr-none"
                      : m.role === "system"
                        ? "bg-destructive/10 text-destructive border border-destructive/20 w-full text-center"
                        : medicalLight
                          ? "bg-white text-neutral-900 border border-neutral-200 rounded-tl-none"
                          : "bg-neutral-800 text-neutral-100 border-neutral-700 rounded-tl-none"
                  }
                `}
                >
                  <div
                    className={`prose prose-sm ${
                      medicalLight ? "" : "prose-invert"
                    } max-w-none break-words`}
                  >
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                </div>
                {m.role === "user" && (
                  <div className="h-8 w-8 rounded-full bg-neutral-100 flex items-center justify-center flex-shrink-0 border border-neutral-200">
                    <User className="h-4 w-4 text-neutral-600" />
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 border border-blue-200">
                  <Bot className="h-4 w-4 text-blue-600" />
                </div>
                <div
                  className={`${
                    medicalLight
                      ? "bg-white border border-neutral-200"
                      : "bg-neutral-800 border-neutral-700"
                  } rounded-2xl rounded-tl-none p-4`}
                >
                  <div className="flex gap-1">
                    <span
                      className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0ms" }}
                    ></span>
                    <span
                      className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                      style={{ animationDelay: "150ms" }}
                    ></span>
                    <span
                      className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    ></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={scrollRef} />
          </div>
        </ScrollArea>

        <div className="flex gap-2 pt-2 border-t">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask follow-up questions..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            size="icon"
            onClick={sendMessage}
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
