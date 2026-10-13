interface ToastProps {
  message: string;
  visible: boolean;
}

export function Toast({ message, visible }: ToastProps) {
  return <div className={`toast-msg${visible ? ' show' : ''}`}>{message}</div>;
}
