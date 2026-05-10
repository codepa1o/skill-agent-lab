const runForm = document.querySelector("[data-run-form]");
const submitButton = document.querySelector("[data-submit-button]");
const formStatus = document.querySelector("#form-status");

if (runForm && submitButton && formStatus) {
  runForm.addEventListener("submit", () => {
    submitButton.disabled = true;
    submitButton.textContent = "运行中...";
    formStatus.textContent = "正在拉取 Skill 并调用模型";
    formStatus.classList.add("running");
  });
}

for (const form of document.querySelectorAll("[data-rerun-form]")) {
  form.addEventListener("submit", () => {
    const button = form.querySelector("button");
    if (button) {
      button.disabled = true;
      button.textContent = "运行中...";
    }
  });
}

const messageList = document.querySelector("[data-message-list]");

if (messageList) {
  messageList.scrollTop = messageList.scrollHeight;
}
