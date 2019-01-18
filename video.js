const { exec } = require('child_process')

ps = process.argv[2] || ''
pr = process.argv[3] || ''

const makeVideo = (s = '', r = '') => {
    pat = s ? s + '_*' : '*'
    pat += r ? '_r' + r + '_*' : ''
    pat += '.png'

    out = s ? s : 'ALL'
    out += r ? '_' + r : ''
    out += '_replay.mp4'

    ep = "cd graphs && ffmpeg -f image2 -r 6 -pattern_type glob -i '" + pat + "' -pix_fmt yuv420p " + out
    console.log('EXEC: ' + ep)
    exec('rm graphs/' + out, (err, stdout, stderr) => {
        // console.log(stdout, stderr)
    })
    exec(ep, (err, stdout, stderr) => {
        console.log('DONE')
        // console.log(stdout, stderr)
    })
}

if (ps == 'logs') {
    console.log('Make Video By Logs:')
    process.exit()
} else {
    makeVideo(ps, pr)
}
