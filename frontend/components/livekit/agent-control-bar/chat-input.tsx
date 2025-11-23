import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { PaperPlaneRightIcon, SpinnerIcon } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';

const MOTION_PROPS = {
  variants: {
    hidden: {
      opacity: 0,
      marginBottom: 0,
    },
    visible: {
      opacity: 1,
      marginBottom: 12,
    },
  },
  initial: 'hidden',
  transition: {
    duration: 0.25,
    ease: 'easeOut',
  },
};

interface ChatInputProps {
  chatOpen: boolean;
  isAgentAvailable?: boolean;
  onSend?: (message: string) => void;
}

export function ChatInput({
  chatOpen,
  isAgentAvailable = false,
  onSend = async () => {},
}: ChatInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isSending, setIsSending] = useState(false);
  const [message, setMessage] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    try {
      setIsSending(true);
      await onSend(message);
      setMessage('');
    } catch (error) {
      console.error(error);
    } finally {
      setIsSending(false);
    }
  };

  const isDisabled =
    isSending || !isAgentAvailable || message.trim().length === 0;

  useEffect(() => {
    if (chatOpen && isAgentAvailable) {
      inputRef.current?.focus();
    }
  }, [chatOpen, isAgentAvailable]);

  return (
    <motion.div
      aria-hidden={!chatOpen} // changed from inert
      {...MOTION_PROPS}
      animate={chatOpen ? 'visible' : 'hidden'}
      className="border-input/50 flex w-full items-start overflow-hidden border-b"
    >
      <form
        onSubmit={handleSubmit}
        className="mb-3 flex grow items-end gap-2 rounded-md pl-1 text-sm"
      >
        <input
          autoFocus
          ref={inputRef}
          type="text"
          value={message}
          disabled={!chatOpen}
          placeholder="Type something..."
          onChange={(e) => setMessage(e.target.value)}
          className="h-8 flex-1 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
        />

        <Button
          size="icon"
          type="submit"
          disabled={isDisabled}
          variant={isDisabled ? 'secondary' : 'primary'}
          title={isSending ? 'Sending...' : 'Send'}
          className="self-start"
        >
          {isSending ? (
            <SpinnerIcon className="animate-spin" weight="bold" />
          ) : (
            <PaperPlaneRightIcon weight="bold" />
          )}
        </Button>
      </form>
    </motion.div>
  );
}
