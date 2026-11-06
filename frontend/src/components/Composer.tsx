import { useEffect, useRef, useState } from "react";

/** 输入框：Enter 发送，Shift+Enter 换行，生成中切换为停止。 */
export function Composer({
  disabled,
  streaming,
  onSend,
  onStop,
}: {
  disabled: boolean;
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={textareaRef}
          value={value}
          rows={1}
          placeholder="问制度、查订单，或让我开一张售后工单"
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          disabled={streaming}
          aria-label="输入问题"
        />
        <div className="composer-bar">
          <p className="hint">Enter 发送 · Shift+Enter 换行</p>
          {streaming ? (
            <button type="button" className="send stop" onClick={onStop}>
              停止
            </button>
          ) : (
            <button
              type="button"
              className="send"
              onClick={submit}
              disabled={!value.trim() || disabled}
            >
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
