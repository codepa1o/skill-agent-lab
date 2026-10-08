const runForm = document.querySelector("[data-run-form]");
const chatForm = document.querySelector("[data-chat-form]");
const submitButton = document.querySelector("[data-submit-button]");
const formStatus = document.querySelector("#form-status");
const messageList = document.querySelector("[data-message-list]");
const conversationPage = document.querySelector("[data-conversation-page]");

const setStatus = (text, running = false) => {
  if (!formStatus) return;
  formStatus.textContent = text;
  formStatus.classList.toggle("running", running);
};

const scrollMessagesToBottom = () => {
  if (messageList) {
    messageList.scrollTop = messageList.scrollHeight;
  }
};

const nowLabel = () =>
  new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date());

const createTypingIndicator = () => {
  const indicator = document.createElement("span");
  indicator.className = "typing-indicator";
  indicator.setAttribute("aria-label", "\u6b63\u5728\u54cd\u5e94");
  return indicator;
};

const createMessageElement = ({ id, role, content = "", status = "complete", animate = false }) => {
  const article = document.createElement("article");
  article.className = `message ${role} ${status}${animate ? " message-enter" : ""}`;
  if (id) {
    article.dataset.messageId = id;
  }

  const header = document.createElement("div");
  header.className = "message-header";

  const author = document.createElement("span");
  author.textContent = role === "user" ? "\u6211" : "Agent";

  const time = document.createElement("span");
  time.textContent = nowLabel();

  const body = document.createElement("pre");
  body.dataset.messageContent = "";
  body.textContent = content;

  header.append(author, time);
  article.append(header, body);

  if (status === "streaming") {
    article.append(createTypingIndicator());
  }

  return article;
};

const setMessageStatus = (element, status) => {
  element.classList.remove("streaming", "complete", "failed");
  element.classList.add(status);
};

const appendError = (messageElement, text) => {
  setMessageStatus(messageElement, "failed");
  messageElement.querySelector(".typing-indicator")?.remove();
  const content = messageElement.querySelector("[data-message-content]");
  if (content) {
    content.textContent = text;
    content.classList.add("error-text");
  }
};

const createTypewriter = (target, speedMs = 12) => {
  let queue = Promise.resolve();
  const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const appendText = async (text) => {
    for (const character of Array.from(text)) {
      target.textContent += character;
      scrollMessagesToBottom();
      await wait(speedMs);
    }
  };

  return {
    enqueue(text) {
      queue = queue.then(() => appendText(text || ""));
      return queue;
    },
    flush() {
      return queue;
    },
  };
};

const readStreamEvents = async (response, handlers) => {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const rawEvent of events) {
      const lines = rawEvent.split("\n");
      const eventName = lines
        .find((line) => line.startsWith("event:"))
        ?.slice("event:".length)
        .trim();
      const dataLines = lines
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice("data:".length).trim());
      if (!eventName || dataLines.length === 0) continue;

      let payload;
      try {
        payload = JSON.parse(dataLines.join("\n"));
      } catch {
        continue;
      }
      const handler = handlers[eventName];
      if (handler) {
        await handler(payload);
      }
    }
  }
};

const openStream = async ({
  question,
  messageId = "",
  appendLocalMessage = true,
  autoStart = false,
  existingUserMessage = null,
}) => {
  if (!chatForm || !submitButton || !messageList) return;

  const formData = new FormData();
  if (messageId) {
    formData.set("message_id", String(messageId));
  } else {
    formData.set("question", question);
  }

  submitButton.disabled = true;
  submitButton.textContent = "\u53d1\u9001\u4e2d";
  setStatus("\u6b63\u5728\u54cd\u5e94", true);

  let userMessage = existingUserMessage;
  let assistantMessage = null;
  if (appendLocalMessage) {
    userMessage = createMessageElement({
      role: "user",
      content: question,
      animate: true,
    });
    assistantMessage = createMessageElement({
      role: "assistant",
      content: "",
      status: "streaming",
      animate: true,
    });
    messageList.append(userMessage, assistantMessage);
    scrollMessagesToBottom();
  } else {
    assistantMessage = createMessageElement({
      role: "assistant",
      content: "",
      status: "streaming",
      animate: autoStart,
    });
    messageList.append(assistantMessage);
    scrollMessagesToBottom();
  }

  const assistantContent = assistantMessage.querySelector("[data-message-content]");
  const typewriter = createTypewriter(assistantContent);
  let completed = false;

  try {
    const response = await fetch(chatForm.dataset.streamAction || chatForm.action, {
      method: "POST",
      body: formData,
    });
    if (!response.ok || !response.body) {
      throw new Error("Request failed: " + response.status);
    }

    await readStreamEvents(response, {
      message(payload) {
        if (payload.role === "user") {
          if (userMessage) {
            userMessage.dataset.messageId = payload.id;
          }
          return;
        }
        assistantMessage.dataset.messageId = payload.id;
      },
      delta(payload) {
        return typewriter.enqueue(payload.content || "");
      },
      async done(payload) {
        await typewriter.flush();
        assistantContent.textContent = payload.content || assistantContent.textContent;
        assistantMessage.querySelector(".typing-indicator")?.remove();
        setMessageStatus(assistantMessage, "complete");
        setStatus(`\u5b8c\u6210 ${payload.latency_ms || 0} ms`);
        completed = true;
      },
      error(payload) {
        appendError(assistantMessage, payload.message || "\u751f\u6210\u5931\u8d25");
        setStatus("\u751f\u6210\u5931\u8d25");
      },
    });
  } catch (error) {
    appendError(assistantMessage, error.message || "\u7f51\u7edc\u5f02\u5e38");
    setStatus("\u8bf7\u6c42\u5931\u8d25");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "\u53d1\u9001";
    scrollMessagesToBottom();
  }
  return completed;
};

if (chatForm && submitButton && messageList) {
  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(chatForm);
    const question = String(formData.get("question") || "").trim();
    if (!question) {
      setStatus("\u8bf7\u8f93\u5165\u95ee\u9898");
      return;
    }

    chatForm.reset();
    await openStream({ question, appendLocalMessage: true });
  });
} else if (runForm && submitButton && formStatus) {
  runForm.addEventListener("submit", () => {
    submitButton.disabled = true;
    submitButton.textContent = "\u8fd0\u884c\u4e2d";
    setStatus("\u6b63\u5728\u52a0\u8f7d Skill \u5e76\u751f\u6210\u56de\u7b54", true);
  });
}

for (const form of document.querySelectorAll("[data-rerun-form]")) {
  form.addEventListener("submit", () => {
    const button = form.querySelector("button");
    if (button) {
      button.disabled = true;
      button.textContent = "\u8fd0\u884c\u4e2d";
    }
  });
}

const consumeAutostartFlag = async () => {
  if (!conversationPage || !chatForm || !messageList) return;
  const autostart = conversationPage.dataset.autostart === "1";
  const messageId = String(conversationPage.dataset.latestUserMessageId || "").trim();
  const conversationId = String(chatForm.dataset.conversationId || "").trim();
  if (!autostart || !messageId || !conversationId) return;

  const storageKey = `skill-agent-autostart:${conversationId}:${messageId}`;
  const userMessage = messageList.querySelector(`[data-message-id="${messageId}"]`);
  const question = userMessage?.querySelector("[data-message-content]")?.textContent?.trim() || "";
  if (!question) return;

  try {
    const storedState = parseAutostartState(window.sessionStorage.getItem(storageKey));
    if (storedState?.status === "done") {
      history.replaceState({}, "", window.location.pathname);
      return;
    }

    if (storedState?.status === "running" && Date.now() - storedState.at < 120000) {
      history.replaceState({}, "", window.location.pathname);
      return;
    }

    window.sessionStorage.setItem(
      storageKey,
      JSON.stringify({ status: "running", at: Date.now() }),
    );
  } catch {
    // Ignore storage access issues and fall back to the URL rewrite below.
  }

  try {
    history.replaceState({}, "", window.location.pathname);
  } catch {
    // Ignore history issues in constrained browser contexts.
  }

  const completed = await openStream({
    question,
    messageId,
    appendLocalMessage: false,
    autoStart: true,
    existingUserMessage: userMessage,
  });

  try {
    if (completed) {
      window.sessionStorage.setItem(
        storageKey,
        JSON.stringify({ status: "done", at: Date.now() }),
      );
    } else {
      window.sessionStorage.removeItem(storageKey);
    }
  } catch {
    // Ignore storage access issues; the URL has already been consumed.
  }
};

const parseAutostartState = (value) => {
  if (!value) return null;
  try {
    const state = JSON.parse(value);
    if (state && typeof state.status === "string") {
      return {
        status: state.status,
        at: Number(state.at || 0),
      };
    }
  } catch {
    return null;
  }
  return null;
};

consumeAutostartFlag().catch(() => {});
scrollMessagesToBottom();

const jobStatus = document.querySelector("[data-job-status]");

if (jobStatus) {
  const status = jobStatus.dataset.jobStatus;
  if (status === "queued" || status === "running") {
    const pollInterval = Number(jobStatus.dataset.pollInterval || "2") * 1000;
    window.setTimeout(() => {
      window.location.reload();
    }, Math.max(1000, pollInterval));
  }
}
