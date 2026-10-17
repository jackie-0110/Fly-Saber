export function PanelLetter({ children }: { children: string }) {
  return (
    <span className="panel-letter" aria-hidden="true">
      {children}
    </span>
  )
}
