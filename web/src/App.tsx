// Layout per spec 14: game view (left), 3D brain (right), controls, fly's-eye inset.
export default function App() {
  return (
    <main className="grid min-h-screen grid-cols-2 gap-2 bg-neutral-950 p-2 text-neutral-100">
      <section className="rounded border border-neutral-800 p-4">Fly Saber scene (game/)</section>
      <section className="rounded border border-neutral-800 p-4">3D brain (brain/)</section>
      <section className="col-span-2 rounded border border-neutral-800 p-4">
        Controls (controls/)
      </section>
    </main>
  )
}
