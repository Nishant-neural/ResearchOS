"use client";

import { FileText, Send, Upload } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";

const messages = [
  {
    role: "assistant",
    text: "Upload a paper, then ask about its methods, equations, architecture, or implementation details.",
  },
  {
    role: "user",
    text: "What problem does this paper solve, and what should I implement first?",
  },
  {
    role: "assistant",
    text: "Day 1 will connect this chat to the FastAPI backend. Retrieval and citations arrive in the next phase.",
  },
];

export function ChatShell() {
  const [input, setInput] = useState("");

  return (
    <main className="flex min-h-screen flex-col bg-background">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div>
            <h1 className="text-lg font-semibold">AI Research OS</h1>
            <p className="text-sm text-muted-foreground">
              Research chat workspace
            </p>
          </div>
          <Button variant="secondary">
            <Upload className="h-4 w-4" aria-hidden="true" />
            Upload PDF
          </Button>
        </div>
      </header>

      <section className="mx-auto grid w-full max-w-7xl flex-1 grid-cols-1 gap-4 px-4 py-4 lg:grid-cols-[280px_1fr]">
        <aside className="rounded-md border bg-white p-4">
          <div className="flex items-center gap-2 text-sm font-medium">
            <FileText className="h-4 w-4 text-primary" aria-hidden="true" />
            Current paper
          </div>
          <div className="mt-4 rounded-md border border-dashed p-4 text-sm text-muted-foreground">
            No PDF uploaded yet.
          </div>
        </aside>

        <div className="flex min-h-[calc(100vh-128px)] flex-col rounded-md border bg-white">
          <div className="flex-1 space-y-4 overflow-y-auto p-4">
            {messages.map((message, index) => (
              <div
                className={
                  message.role === "user"
                    ? "ml-auto max-w-2xl rounded-md bg-primary px-4 py-3 text-sm text-primary-foreground"
                    : "max-w-2xl rounded-md bg-muted px-4 py-3 text-sm"
                }
                key={`${message.role}-${index}`}
              >
                {message.text}
              </div>
            ))}
          </div>

          <form
            className="flex gap-2 border-t p-4"
            onSubmit={(event) => {
              event.preventDefault();
              setInput("");
            }}
          >
            <input
              aria-label="Ask a research question"
              className="h-10 flex-1 rounded-md border bg-background px-3 text-sm outline-none ring-offset-background placeholder:text-muted-foreground focus:ring-2 focus:ring-ring"
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask about the paper..."
              value={input}
            />
            <Button size="icon" type="submit">
              <Send className="h-4 w-4" aria-hidden="true" />
              <span className="sr-only">Send</span>
            </Button>
          </form>
        </div>
      </section>
    </main>
  );
}
