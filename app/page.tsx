import Link from 'next/link'

const FEATURES = [
  {
    icon: '🎯',
    title: 'Joint Tracking',
    desc: 'MediaPipe AI marks 33 body landmarks, shoulders, elbows, wrists, knees, on every frame.',
  },
  {
    icon: '🌊',
    title: 'Swing Path Trails',
    desc: 'See your racket-hand motion path traced in real-time as a smooth Bezier curve.',
  },
  {
    icon: '👁️',
    title: 'Toggle Any Joint',
    desc: 'Show or hide individual joint groups to focus on what matters for your shot.',
  },
  {
    icon: '📌',
    title: 'Save a Baseline',
    desc: 'Mark your best-day swing as a baseline. Compare every future session against it.',
  },
  {
    icon: '🤖',
    title: 'AI Coaching',
    desc: 'Claude reads your joint angles vs your best day and tells you what held up and what drifted.',
  },
  {
    icon: '📐',
    title: 'Side by Side',
    desc: 'Synchronized video playback. Scrub both swings in lockstep at any speed.',
  },
]

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-900/30 via-transparent to-blue-900/20" />
        <div className="relative max-w-5xl mx-auto px-4 pt-24 pb-20 text-center">
          <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-4 py-1.5 text-emerald-400 text-sm font-medium mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            AI-powered swing analysis
          </div>

          <h1 className="text-5xl md:text-7xl font-black text-white leading-tight mb-6">
            Beat your
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
              last swing.
            </span>
          </h1>

          <p className="text-white/60 text-xl max-w-2xl mx-auto mb-10 leading-relaxed">
            AI pose tracking for every swing you record. See your technique get sharper,
            more consistent, more powerful — week over week.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/analyze"
              className="px-8 py-4 bg-emerald-500 hover:bg-emerald-400 text-white font-bold rounded-2xl text-lg transition-all hover:scale-105 hover:shadow-lg hover:shadow-emerald-500/25"
            >
              Analyze My Swing
            </Link>
            <Link
              href="/baseline"
              className="px-8 py-4 bg-white/10 hover:bg-white/15 text-white font-bold rounded-2xl text-lg transition-all border border-white/10"
            >
              My Baselines
            </Link>
          </div>
        </div>
      </section>

      {/* Feature grid */}
      <section className="max-w-6xl mx-auto px-4 py-20">
        <h2 className="text-3xl font-bold text-white text-center mb-12">
          Everything you need to level up
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="rounded-2xl border border-white/5 bg-white/[0.03] p-6 hover:bg-white/[0.06] transition-colors"
            >
              <div className="text-3xl mb-3">{f.icon}</div>
              <h3 className="text-white font-semibold text-lg mb-2">{f.title}</h3>
              <p className="text-white/50 text-sm leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="border-t border-white/5 py-20">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-white mb-12">How it works</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { n: '1', label: 'Upload', desc: 'Drop your swing video (MP4/MOV)' },
              { n: '2', label: 'Track', desc: 'AI extracts 33 joint landmarks from every frame' },
              { n: '3', label: 'Pin a baseline', desc: 'Mark your best swing as the reference' },
              { n: '4', label: 'Beat it', desc: 'Upload future swings and see what changed' },
            ].map((step) => (
              <div key={step.n} className="flex flex-col items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-black text-xl">
                  {step.n}
                </div>
                <h3 className="text-white font-semibold">{step.label}</h3>
                <p className="text-white/50 text-sm">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-white/5 py-20 text-center">
        <div className="max-w-xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to see the difference?
          </h2>
          <p className="text-white/50 mb-8">
            No sign-up required. Upload your video and get instant feedback.
          </p>
          <Link
            href="/analyze"
            className="inline-block px-8 py-4 bg-emerald-500 hover:bg-emerald-400 text-white font-bold rounded-2xl text-lg transition-all hover:scale-105"
          >
            Start Analyzing Free
          </Link>
        </div>
      </section>
    </div>
  )
}
